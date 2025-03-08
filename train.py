import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score, MeanSquaredError, Metric
from torch import nn
from typing import Callable, Type
import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error

from data_handler import AmeriFLUXDataset, prepare_data, Site, get_site_vars
from model_class import MODEL_HYPERPARAMETERS, XGBoost, RandomForest, NEPModel

HYPERPARAMETER_NAMES = {
    'epochs' : {'title' : 'Epochs', 'abbreviation' : "EP"},
    'lr' : {'title' : 'Learning Rate', 'abbreviation' : 'LR'},
    'batch_size' : {'title' : 'Batch Size', 'abbreviation' : "BS"},
    'activation_fn' : {'title' : 'Activation Function', 'abbreviation' : "F"},
    'layer_dims' : {'title' : 'Layer Dimensions', 'abbreviation' : 'LD'},
    'stat_interval' : {'title' : 'Rolling Statistics Interval', 'abbreviation' : 'SI'},
    'sequence_length' : {'title' : 'Time Series Sequence Length', 'abbreviation' : 'SL'},
    'hidden_state_size' : {'title' : 'Hidden State Size', 'abbreviation' : 'H'},
    'num_layers' : {'title' : 'Number Hidden Layers', 'abbreviation' : 'L'},
    'dropout' : {'title' : 'Dropout', 'abbreviation' : 'D'},
    'n_estimators' : {'title': 'Number Estimators', 'abbreviation': 'NE'}
}

##### Base Train, Test, Eval functions #######
def train(dataloader : DataLoader,
          model : nn.Module,
          loss_fn : nn.Module, # not necessarily mseloss
          optimizer : torch.optim.Optimizer,
          device,
          skip_training_curve = False) -> None:
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Get predictions and loss value
        pred = model(X.to(device))
        loss : torch.Tensor = loss_fn(pred, y.to(device))
        # backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader : DataLoader, model : nn.Module, loss_fn : nn.Module, device) -> None:
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f}")
    return test_loss

def eval(dataloader : DataLoader, model : nn.Module, metric_fn : Metric, device) -> float:
    size = len(dataloader.dataset)
    model.eval()

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X.to(device))
            metric_fn.update(input=pred, target=y.to(device))
            
    print(f"Metric value: {metric_fn.compute().item():>8f}")
    return metric_fn.compute().item()

def train_test(train_dataloader, test_dataloader, model, epochs, loss_fn, optimizer, device, skip_curve = False, context=None):
    history = []
    for t in range(epochs):
        print(f"Epoch {t+1}")
        train(train_dataloader, model, loss_fn, optimizer, device)
        train_loss = test(train_dataloader, model, loss_fn, device)
        test_loss = test(test_dataloader, model, loss_fn, device)

        # compute the r squared value for this model
        r2metric = R2Score(device=device)
        r2 = eval(test_dataloader, model, r2metric, device)
        history.append({'epoch': t+1, 'train_loss': train_loss, 'test_loss': test_loss, 'r2': r2})

    if not skip_curve:
        plt.clf()
        epochs = list(range(1, epochs+1))
        train_loss = [e['train_loss'] for e in history]
        test_loss = [e['test_loss'] for e in history]
        plt.plot(epochs, train_loss, label=f'Training Loss ({loss_fn.__class__.__name__})', c='b')
        plt.plot(epochs, test_loss, label=f'Test Loss ({loss_fn.__class__.__name__})', c='r')
        plt.ylim((0, 14))
        title = f'{model.__class__.__name__} training curve'
        if context is not None:
            title += f' ({context})'
        plt.title(title)
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        #plt.show()
        dt_str = fmt_date_string()
        plt.savefig(f'images/training_curve-{dt_str}::{model.__class__.__name__}.png')
    return history


###### Helper Functions that encapsulate the common training-testing procedures #######

# Train with K-Fold cross validation
def train_kfold(num_folds : int,
                model_class : Type[nn.Module],
                lr : float, bs : int, epochs : int, loss_fn : nn.Module,
                train_data : AmeriFLUXDataset,
                device, num_features, weight_decay=0,
                **model_kwargs) -> float:

    r2_results = {}
    loss_results = {}
    sklearn_model = model_class.__name__ in ['XGBoost', 'RandomForest']

    # use our own fold indexing
    for fold in range(num_folds):
        print(f"Fold {fold+1}: ")
        # set up next model
        if sklearn_model:
            model = model_class(lr=lr, **model_kwargs)
        else:
            model : nn.Module = model_class(num_features, batch_size=bs, **model_kwargs).to(device)
        if fold==0:
            print(model)
        # start with the last year as the first validation year and work our way back with each fold
        train_idx, test_idx = train_data.get_train_test_idx(fold)
        if train_idx is None:
            print("No more one-year folds can be done, ending K-Fold cross validation")
            break

        if sklearn_model:
            X = train_data.get_X()
            y = train_data.get_y()
            X_train = [X[idx] for idx in train_idx]
            y_train = [y[idx] for idx in train_idx]
            X_test = [X[idx] for idx in test_idx]
            y_test = [y[idx] for idx in test_idx]
            model.fit(X_train, y_train)

            # compute the r squared value for this model
            r2_results[fold] = r2_score(y_test, model(X_test))
            loss_results[fold] = mean_squared_error(y_test, model(X_test))
        else:
            # set up dataloaders
            # is there a better way to samnple?
            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)

            train_loader = DataLoader(train_data, batch_size=bs, sampler=train_subsampler, drop_last=True)
            test_loader = DataLoader(train_data, batch_size=1, sampler=test_subsampler)

            # Using SGD here but could also do Adam or others
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

            history = train_test(train_loader, test_loader, model, epochs, loss_fn, optimizer, device,context='K-Fold', skip_curve=True)
            r2_results[fold] = history[-1]['r2']
            loss_results[fold] = history[-1]['train_loss']

    # display r squared results
    print("K-Fold Cross Validation Results")
    num_folds_actual = len(r2_results.items())
    r2_total = 0
    for i in range(num_folds_actual):
        print(f"Fold {i}: r2={r2_results[i]:.3f}, {loss_fn.__class__.__name__}={loss_results[i]:.3f}")
    sum_r2 = sum(r2_results.values())
    avg_r2 = sum_r2/num_folds_actual
    sum_loss = sum(loss_results.values())
    avg_loss = sum_loss/num_folds_actual
    stddev_r2 = (sum([(r2-avg_r2)**2 for r2 in r2_results.values()])/num_folds_actual)**0.5
    stddev_loss = (sum([(loss-avg_loss)**2 for loss in loss_results.values()])/num_folds_actual)**0.5
    print(f"Average r-squared: {avg_r2}; stddev : {stddev_r2}")
    print(f"Average loss: {avg_loss}; stddev : {stddev_loss}")


    # Determine the lower bound of the 95% confidence interval
    # t_table = {1: 100, 2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.37, 9: 2.31, 10: 2.26}
    # t_score = t_table[num_folds_actual]
    # ci_low = avg_r2 - t_score*(stddev/num_folds_actual**0.5)
    # ci_high =avg_r2 + t_score*(stddev/num_folds_actual**0.5)
    # print(f"95% confidence interval: {ci_low} - {ci_high}")

    return avg_r2, avg_loss

def train_hparam(model_class : Type[NEPModel] | Type[XGBoost] | Type[RandomForest], site, input_columns, **kwargs) -> tuple[list[NEPModel | XGBoost | RandomForest], dict[str, None], list[dict[str, None]]]:
    model_name = model_class.__name__

    # Accepted kwargs
    num_folds = kwargs.get('num_folds', 5)
    num_models = kwargs.get('num_models', 1)
    
    sklearn_model = model_name=='XGBoost' or model_name=='RandomForest'

    model_hparams = MODEL_HYPERPARAMETERS[model_name]
    
    skip_eval = kwargs.get('skip_eval', False)
    flatten = kwargs.get('flatten', False)

    time_series = True if model_name in ['RNN', 'LSTM'] or 'sequence_length' in kwargs.keys() else False
    
    # 2. Initialize the model
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    # Construct the test combinations to test
    data_hparams = ['sequence_length', 'stat_interval']
    hparam_lists = []
    data_hparam_lists = []
    for hparam, default in model_hparams.items():
        # Collect the desired set of values/single value for the hparam from kwargs
        hparam_kwarg = kwargs.get(hparam, default)
        # Ensure it is in a list
        _hparam_list = hparam_kwarg if isinstance(hparam_kwarg, list) else [hparam_kwarg]
        # Build a list of the form [{hparam: val1}, {hparam: val2}, ...]
        hparam_list = []
        for hparam_val in _hparam_list:
            hparam_list.append({hparam: hparam_val})
        # add to data hparams or model hparams
        if hparam in data_hparams:
            kwargs.pop(hparam, None)
            data_hparam_lists.append(hparam_list)
        else:
            hparam_lists.append(hparam_list)

    data_candidates = []
    # key_values is a list of individual dictionaries with hyperparameter names and values
    for key_values in itertools.product(*data_hparam_lists):
        data_candidate = {}
        for key_value in key_values:
            data_candidate.update(key_value)
        data_candidates.append(data_candidate)

    candidates = []
    for key_values in itertools.product(*hparam_lists):
        candidate = {}
        for key_value in key_values:
            candidate.update(key_value)
        candidates.append(candidate)

    history = []
    best = {}
    data_best = {}
    if len(candidates)*len(data_candidates) == 1:
        print("Only one candidate hyperparameter configuration, skipping K-Fold cross validation")
        best = candidates[0]
        data_best = data_candidates[0]
        best['r2'] = np.nan
        best['loss'] = np.nan
        history = [best | data_best | {'train_size' : np.nan}]

    else:
        print(f"Running K-Fold Cross Validation on {len(candidates)*len(data_candidates)} different hyperparameter configurations")
        max_r2 = -np.inf
        min_loss = np.inf
        for data_candidate in data_candidates:
            hparam_print = "Hyperparameters:\n"
            for key, val in data_candidate.items():
                hparam_print += f"{HYPERPARAMETER_NAMES[key]['title']}: {val}\n"
            
            if time_series:
                train_data, _ = prepare_data(site, input_columns, **data_candidate, **kwargs)
            else:
                train_data, _ = prepare_data(site, input_columns, **data_candidate, **kwargs)
            if train_data == None or len(train_data) == 0: # or train_data.get_num_years() <= 1:
                print("Training set does not have enough data. Skipping this candidate...")
                continue
            num_features = len(input_columns)*(3 if not time_series and 'stat_interval' in data_candidate.keys() and data_candidate['stat_interval'] is not None else 1)
            for candidate in candidates:
                candidate_suffix = ''
                for key, val in candidate.items():
                    candidate_suffix += f"{HYPERPARAMETER_NAMES[key]['title']}: {val}\n"
                print(hparam_print + candidate_suffix)
                candidate_subset = dict(candidate).copy()
                candidate_subset.pop('lr', None)
                candidate_subset.pop('batch_size', None)
                candidate_subset.pop('epochs', None)
                r2, loss = train_kfold(min(num_folds, train_data.get_num_years()), model_class,
                                candidate.get('lr', -1),
                                candidate.get('batch_size', -1),
                                candidate.get('epochs', -1),
                                loss_fn, train_data, device, num_features*data_candidate['sequence_length'] if time_series and flatten else num_features,
                                **candidate_subset)
                history.append(candidate | data_candidate | {'train_size' : len(train_data)})
                history[-1]['r2'] = r2
                history[-1]['loss'] = loss
                #history[-1]['ci_low'] = ci_low
                if r2 > max_r2:
                    best = candidate
                    data_best = data_candidate
                    max_r2 = r2
                    min_loss = loss
        if max_r2 == -np.inf:
            print("No candidates succeeded (likely all datasets were empty)")
            return [], {'r2': -np.inf, 'loss' : np.inf}, None
        best['r2'] = max_r2
        best['loss'] = min_loss

    # 3. Train with final hparam selections
    models = []
    print(f"Training the best performing model on the entire training set {num_models} times")
    num_features = len(input_columns)*(3 if not time_series and 'stat_interval' in data_best.keys() and data_best['stat_interval'] is not None else 1)
    for i in range(num_models):
        if time_series:
            train_data, eval_data = prepare_data(site, input_columns, **data_best, **kwargs)
        else:
            train_data, eval_data = prepare_data(site, input_columns, **data_best, **kwargs)
        if sklearn_model:
            model : XGBoost | RandomForest = model_class(**best)
            X = train_data.get_X()
            y = train_data.get_y()
            model.fit(X, y)
        else:
            model : NEPModel = model_class(num_features*data_best['sequence_length'] if time_series and flatten else num_features, **best).to(device)
            train_loader = DataLoader(train_data, batch_size=best['batch_size'], drop_last=True) # as long as we are using Adam, maybe want to drop the last batch if it is smaller than the rest
            eval_loader = DataLoader(eval_data, batch_size=64)
            optimizer = torch.optim.SGD(model.parameters(), lr=best['lr'])
            final_model_history = train_test(train_loader, eval_loader, model, best['epochs'], loss_fn, optimizer, device, context='Best Model')
        models.append(model)


    if skip_eval:
        return models, best | data_best, history
    # visualize the training performance of the final model across time
    dt_str = fmt_date_string()
    plot_predictions(f'images/trainplot-{dt_str}::{model_name}-{num_folds}fold', models, train_data, best, device)

    return models, best | data_best, history

def plot_predictions(file : str, models : list[object], data : AmeriFLUXDataset, hyperparams : dict, device : str, train=True, smooth=False, smooth_weight=0.5):
    month_ticks = {1: [1, 14, 28],
                   2: [11, 25],
                   3: [11, 25],
                   4: [8, 22],
                   5: [6, 20],
                   6: [10, 24],
                   7: [8, 22],
                   8: [5,  19],
                   9: [2, 16, 30],
                   10: [14, 28],
                   11: [11, 25],
                   12: [9, 23, 31]
                   }
    
    def smooth_curve(Y : list[float], w : float) -> np.ndarray[float]:
        last = Y[0]
        smoothed_Y = []
        for y in Y:
            smoothed_y = last*w + (1-w)*y
            smoothed_Y.append(smoothed_y)
            last = smoothed_y
        return np.array(smoothed_Y)
    smooth_fn = smooth_curve if smooth else (lambda Y, w: Y)
    _, test_idx = data.get_train_test_idx(0)
    # option: look at just the final ~6 months
    test_idx = test_idx[-len(test_idx):]
    test_subsampler = SubsetRandomSampler(test_idx)
    dates = data.get_dates(idx_range=test_idx)
    year = dates[0].year
    month_labels =["Jan-April", "May-Aug", "Sept-Dec"]
    month_nums = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    # split plots up into 3 four-month subintervals: Jan-April, May-Aug, Sept-Dec
    x = [[d.date() for d in dates if d.month in _month_nums] for _month_nums in month_nums]
    _y = [data.get_y()[idx] for idx in test_idx]
    # split y (targets) on the same points as x (dates)
    y = [_y[:len(x[0])], _y[len(x[0]):len(x[0])+len(x[1])], _y[len(x[0])+len(x[1]):]]
    _X = np.array([data.get_X()[idx] for idx in test_idx])
    # split X (inputs) on the same points as x (dates)
    X = [_X[:len(x[0])], _X[len(x[0]):len(x[0])+len(x[1])], _X[len(x[0])+len(x[1]):]]
    # Pytorch models will want a Tensor over a numpy array
    X_tensor = [torch.tensor(_X, device=device, dtype=torch.float32) for _X in X]
    for _month_nums, month_label, _x, _X, _X_tensor, _y in zip(month_nums, month_labels, x, X, X_tensor, y):
        y_predictions = []
        for model in models:
            if model.__class__.__name__ in ['XGBoost', 'RandomForest']:
                y_pred = model(_X)
                y_predictions.append(y_pred)
            else:
                _y_pred : torch.Tensor = model(_X_tensor)
                y_pred = [a[0] for a in _y_pred.detach().cpu().numpy()]
                y_predictions.append(y_pred)
        y_predictions = np.array(y_predictions)
        y_pred_avg = y_predictions.transpose().mean(axis=1)
        y_pred_var = y_predictions.transpose().var(axis=1)

        plt.clf()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(_x, smooth_fn(_y, smooth_weight), label='Actual NEE')

        if len(models) > 1:
            plt.plot(_x, smooth_fn(y_pred_avg, smooth_weight), label='Average predicted NEE')
            for y_prediction in y_predictions:
                plt.scatter(_x, smooth_fn(y_prediction, smooth_weight), c='b', s=3.0, alpha=0.5)
            plt.fill_between(_x, smooth_fn(y_pred_avg, smooth_weight) - smooth_fn(y_pred_var, smooth_weight), smooth_fn(y_pred_avg, smooth_weight) + smooth_fn(y_pred_var, smooth_weight), alpha=0.2)
        else:
            plt.plot(_x, smooth_fn(y_predictions.flatten(), smooth_weight), label='Predicted NEE')

        # xticks should fall on first day of each week, for a total of 4 months*4 weeks= 16 weeks?
        plt.xticks([datetime.datetime(year=year, month=m, day=d) for m in _month_nums for d in month_ticks[m]])
        plt.yticks(list(range(-8, 2, 2)))
        plt.ylim((-10, 2))
        plt.gcf().autofmt_xdate()
        plt.ylabel("NEE")
        plt.legend()
        plt.title(f"NEE Model Predictions on final year of {'training' if train else 'evaluation'} data {month_label}")
        subtitle = f"{type(model).__name__}"
        for hparam, value in hyperparams.items():
            if hparam=='r2' or hparam=='loss':
                continue
            abbr = HYPERPARAMETER_NAMES[hparam]['abbreviation']
            if isinstance(value, float):
                val_fs = f'{value:.2f}'
            elif isinstance(value, Callable):
                val_fs = f'{value.__name__}'
            else:
                val_fs = f'{value}'
            subtitle += f' | {abbr}: {val_fs}'
        plt.suptitle(subtitle)
        #dt = datetime.datetime.now()
        #name = model.__class__.__name__
        plt.savefig(file.rstrip('.png') + f'-{month_label}.png')

### We are likely not going down this road anymore
# def feature_pruning(model_class : Type[nn.Module], site : Site, **kwargs) -> list[str]:
#     input_columns : list[str] = kwargs.get('input_columns', None)
#     stat_interval = kwargs.get('stat_interval', False)
#     # if no columns are given, start with the full set of variables
#     if input_columns is None:
#         input_columns = get_site_vars(site)
#     # exclude required columns from the feature pruning process
#     required_cols = ['TIMESTAMP_START','NEE_PI','NEE_PI_F','USTAR']
#     for required_col in required_cols:
#         if required_col in input_columns:
#             input_columns.remove(required_col)
#     # PPFD_IN is required
#     if 'PPFD_IN' not in input_columns:
#         input_columns.append('PPFD_IN')
#     # TODO: prune until we no longer improve r-squared?
#     # or prune a specific # of features
#     # or test every combination of columns O(2^n) and pick the best one
#     # get initial model performance
#     history = {}
#     kwargs.pop('input_columns', None)
#     _, results, _ = train_hparam(model_class, site, input_columns, eval_years=1, skip_eval=True, **kwargs)
#     max_r2 = results['r2']
#     # history indexed by # of pruned vars
#     history[0] = {'pruned_col': None, 'r2': results['r2']}
#     pruned_columns = input_columns
#     pruning = True
#     cols_pruned = 0
#     while pruning:
#         new_columns = pruned_columns.copy()
#         pruned_column = None
#         for pruned_idx in range(len(pruned_columns)):
#             # do not prune PPFD_IN
#             if pruned_columns[pruned_idx] == 'PPFD_IN':
#                 continue
#             print(f"Training model without {pruned_columns[pruned_idx]}")
#             _candidate_columns = pruned_columns[0:pruned_idx] + pruned_columns[pruned_idx+1:]# if stat interval is included then 

#             _, results, _ = train_hparam(model_class, site, _candidate_columns, skip_eval=True, **kwargs)
#             _r2 = results['r2']
#             if _r2 > max_r2:
#                 new_columns = _candidate_columns
#                 pruned_column = pruned_columns[pruned_idx]
#                 max_r2 = _r2
#         # if we did not find a better set of columns, then end the pruning process
#         if len(new_columns) == len(pruned_columns):
#             pruning=False
#         else:
#             cols_pruned += 1
#             pruned_columns = new_columns
#             history[cols_pruned] = {'pruned_col': pruned_column, 'r2': max_r2}

#     # once pruning is complete, write the history to a file
#     dt = datetime.datetime.now()
#     with open(f"results/pruning_results-{dt.year}-{dt.month:02}-{dt.day:02}::{dt.hour:02}:{dt.minute:02}:{dt.second:02}.md", 'w') as f:
#         f.write('### Feature Pruning History\n\n')
#         f.write('| # Pruned Columns | Last Pruned Column | Average R-Squared |\n')
#         f.write('| --- | --- | --- |\n')
#         for num_pruned in range(len(history.keys())):
#             f.write(f'| {num_pruned} | {history[num_pruned]["pruned_col"]} | {history[num_pruned]["r2"]:.4f} |\n')

#         f.write('\n')
#         f.write('## Best Performing feature set:\n\n')
#         for col in pruned_columns:
#             f.write(f'{col}\n\n')
#     return pruned_columns
    

"""
Takes a model_class, site and input columns, and any data/model arguments.
Executes hyperparameter tuning and best-model(s) training via train_hparam.
Evaluates each best-model on the evaluation set and training set.
Records the results of train_hparam and the evaluation, and returns the mean r squared and MSE for both eval and train set.
"""
def train_test_eval(model_class : Type[nn.Module], site, input_columns, **kwargs) -> tuple[float, float, float, float]:
    num_folds = kwargs.get('num_folds', 5)
    #time_series = kwargs.get('time_series', False)
    skip_eval = kwargs.get('skip_eval', False)
    model_name = model_class.__name__
    sklearn_model = model_name in ['XGBoost', 'RandomForest']
    time_series = True if model_name in ['LSTM', 'RNN'] or 'sequence_length' in kwargs.keys() else False

    # Perform grid search with k-fold cross validation to optimize the hyperparameters
    final_models, hparams, history = train_hparam(model_class, site, input_columns, **kwargs)
    if len(final_models) == 0:
        print("train_hparam failed to train any models")
        return -np.inf

    # Evaluate the final models on the evaluation set
    data_kwargs = kwargs.copy()
    data_kwargs.update(hparams)
    if time_series:
        kwargs.pop('sequence_length', None)
        train_data, eval_data = prepare_data(site, input_columns, **data_kwargs)
    else:
        kwargs.pop('stat_interval', None)
        train_data, eval_data = prepare_data(site, input_columns, **data_kwargs)

    r2_evals = []
    r2_train = []
    mse_evals = []
    mse_train = []
    y = eval_data.get_y()
    y_train = train_data.get_y()
    for model in final_models:
        if sklearn_model:
            train_y_pred = model(train_data.get_X())
            y_pred = model(eval_data.get_X())
            r2_evals.append(r2_score(y, y_pred))
            r2_train.append(r2_score(y_train, train_y_pred))
            mse_evals.append(mean_squared_error(y, y_pred))
            mse_train.append(mean_squared_error(y_train, train_y_pred))
            device = None
        else:
            train_loader = DataLoader(train_data, batch_size=64)
            eval_loader = DataLoader(eval_data, batch_size=1)
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            r2_metric = R2Score(device=device)
            mse_metric = MeanSquaredError(device=device)
            r2_evals.append(eval(eval_loader, model, r2_metric, device))
            r2_train.append(eval(train_loader, model, r2_metric, device))
            mse_evals.append(eval(eval_loader, model, mse_metric, device))
            mse_train.append(eval(train_loader, model, mse_metric, device))

    if skip_eval:
        return np.mean(r2_evals), np.mean(mse_evals), np.mean(r2_train), np.mean(mse_train)

    dt_str = fmt_date_string()
    # plot evaluation performance
    plot_predictions(f'images/evalplot-{dt_str}::{model_name}-{num_folds}fold', final_models, eval_data, hparams, device, train=False)

    # write the results to an output file
    with open(f"results/training_results-{dt_str}::{model_name}-{num_folds}fold.md", 'w') as f:
        f.write("### K-Fold Cross-Validation History\n\n")
        f.write(f'## {model_name}\n\n')
        f.write(f"Folds: {num_folds}\n\n")

        table_head = '|'
        head_border = '|'
        for hparam in hparams.keys():
            if hparam=='r2' or hparam=='loss':
                continue
            title = HYPERPARAMETER_NAMES[hparam]['title']
            table_head += f' {title} |'
            head_border += ' --- |'
        #table_head += ' Average R-Squared |'
        #head_border += ' --- |'

        f.write(f'{table_head} Training Set Size | Average R-Squared | Average MSE Loss |\n')
        f.write(f'{head_border} --- | --- | --- |\n')
        for candidate in history:
            row = '|'
            for hparam in hparams.keys():
                if hparam=='r2' or hparam=='loss':
                    continue
                value = candidate[hparam]
                if isinstance(value, float):
                    row += f' {value:.3f} |'
                elif isinstance(value, Callable):
                    row += f' {value.__name__} |'
                else:
                    row += f' {value} |'
            row += f' {candidate["train_size"]} | {candidate["r2"]:.3f} | {candidate["loss"]:.3f} |'
            f.write(f"{row}\n")
            
        f.write('\n')
        f.write("### Best Model Evaluation\n\n")
        model_str_lines = str(final_models[0]).splitlines()
        f.write('~~~\n')
        for line in model_str_lines[:-1]:
            f.write(f'{line}\n\n')
        f.write(f'{model_str_lines[-1]}\n~~~')
        f.write('\n\n')

        f.write(f'{table_head} Evaluation R-Squared |  Evaluation MSE |\n')
        f.write(f'{head_border} --- | --- |\n')

        row = '|'
        for hparam in hparams.keys():
            if hparam=='r2' or hparam=='loss':
                continue
            value = hparams[hparam]
            if isinstance(value, float):
                row += f' {value:.3f} |'
            elif isinstance(value, Callable):
                row += f' {value.__name__} |'
            else:
                row += f' {value} |'
        f.write(f"{row} {np.mean(r2_evals):.3f} | {np.mean(mse_evals):.3f} |\n")
    
    return np.mean(r2_evals), np.mean(mse_evals), np.mean(r2_train), np.mean(mse_train)

def fmt_date_string() -> str:
    dt = datetime.datetime.now()
    dt_str = f'{dt.year}-{dt.month:02}-{dt.day:02}-{dt.hour:02}:{dt.minute:02}:{dt.second:02}'
    return dt_str