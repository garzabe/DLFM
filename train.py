import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score
from torch import nn
from typing import Callable, Type
import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score

from data_handler import AmeriFLUXDataset, prepare_data, Site, get_site_vars
from model_class import MODEL_HYPERPARAMETERS

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
          loss_fn : Callable, # not necessarily mseloss
          optimizer : torch.optim.Optimizer,
          device) -> None:
    
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

def test(dataloader : DataLoader, model : nn.Module, loss_fn : nn.MSELoss, device) -> None:
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

def eval(dataloader : DataLoader, model : nn.Module, metric_fn, device) -> float:
    size = len(dataloader.dataset)
    model.eval()

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X.to(device))
            metric_fn.update(input=pred, target=y.to(device))
            
    print(f"Metric value: {metric_fn.compute().item():>8f}")
    return metric_fn.compute().item()


###### Helper Functions that encapsulate the common training-testing procedures #######

# Train with K-Fold cross validation
def train_kfold(num_folds : int,
                model_class : Type[nn.Module],
                lr : float, bs : int, epochs : int, loss_fn : Callable,
                train_data : AmeriFLUXDataset,
                device, num_features,
                **model_kwargs) -> float:

    r2_results = {}
    sklearn_model = model_class.__name__ in ['XGBoost', 'RandomForest']

    # use our own fold indexing
    for fold in range(num_folds):
        print(f"Fold {fold+1}: ")
        # set up next model
        if sklearn_model:
            model = model_class(learning_rate=lr, **model_kwargs)
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
        else:
            # set up dataloaders
            # is there a better way to samnple?
            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)

            train_loader = DataLoader(train_data, batch_size=bs, sampler=train_subsampler)
            test_loader = DataLoader(train_data, batch_size=64, sampler=test_subsampler)

            # Using SGD here but could also do Adam or others
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            for t in range(epochs):
                print(f"Epoch {t+1}")
                train(train_loader, model, loss_fn, optimizer, device)
                test(test_loader, model, loss_fn, device)

            # compute the r squared value for this model
            r2metric = R2Score(device=device)
            r2_results[fold] = eval(test_loader, model, r2metric, device)

    # display r squared results
    print("K-Fold Cross Validation Results")
    num_folds_actual = len(r2_results.items())
    total = 0
    for key, value in r2_results.items():
        print(f"Fold {key}: r2={value:.4f}")
        total += value
    avg_r2 = total/num_folds_actual
    stddev = (sum([(r2-avg_r2)**2 for r2 in r2_results.values()])/num_folds_actual)**0.5
    print(f"Average r-squared: {avg_r2}; stddev : {stddev}")

    # Determine the lower bound of the 95% confidence interval
    t_table = {1: 100, 2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.37, 9: 2.31, 10: 2.26}
    t_score = t_table[num_folds_actual]
    ci_low = avg_r2 - t_score*(stddev/num_folds_actual**0.5)
    ci_high =avg_r2 + t_score*(stddev/num_folds_actual**0.5)
    print(f"95% confidence interval: {ci_low} - {ci_high}")

    return avg_r2, ci_low



def train_hparam(model_class : Type[nn.Module], **kwargs) -> tuple[list[nn.Module], dict[str, None], list[dict[str, None]]]:
    model_name = model_class.__name__

    # Accepted kwargs
    num_folds = kwargs.get('num_folds', 5)
    input_columns = kwargs.get('input_columns', [])
    site = kwargs.get('site', Site.Me2)
    num_models = kwargs.get('num_models', 1)
    
    sklearn_model = model_name=='XGBoost' or model_name=='RandomForest'

    model_hparams = MODEL_HYPERPARAMETERS[model_name]
    
    skip_eval = kwargs.get('skip_eval', False)
    time_series = kwargs.get('time_series', False)
    flatten = kwargs.get('flatten', False)
    

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

    print(f"Running K-Fold Cross Validation on {len(candidates)*len(data_candidates)} different hyperparameter configurations")
    history = []
    best = None
    data_best = None
    max_r2 = -np.inf
    for data_candidate in data_candidates:
        hparam_print = "Hyperparameters:\n"
        for key, val in data_candidate.items():
            hparam_print += f"{HYPERPARAMETER_NAMES[key]['title']}: {val}\n"
        
        if time_series:
            train_data, _ = prepare_data(site, **data_candidate, **kwargs)
        else:
            train_data, _ = prepare_data(site, **data_candidate, **kwargs)
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
            r2, ci_low = train_kfold(min(num_folds, train_data.get_num_years()), model_class,
                            candidate.get('lr', -1),
                            candidate.get('batch_size', -1),
                            candidate.get('epochs', -1),
                            loss_fn, train_data, device, num_features*data_candidate['sequence_length'] if time_series and flatten else num_features,
                            **candidate_subset)
            history.append(candidate | data_candidate | {'train_size' : len(train_data)})
            history[-1]['r2'] = r2
            #history[-1]['ci_low'] = ci_low
            if ci_low > max_r2: # to account for folds with different # of years, we use confidence intervals
                best = candidate
                data_best = data_candidate
                max_r2 = ci_low
    if max_r2 == -np.inf:
        print("No candidates succeeded (likely all datasets were empty)")
        return [], {'r2': -np.inf}, None
    best['r2'] = max_r2

    # 3. Train with final hparam selections
    models = []
    print(f"Training the best performing model on the entire training set {num_models} times")
    num_features = len(input_columns)*(3 if not time_series and 'stat_interval' in data_best.keys() and data_best['stat_interval'] is not None else 1)
    for i in range(num_models):
        if time_series:
            train_data, _ = prepare_data(site, **data_best, **kwargs)
        else:
            train_data, _ = prepare_data(site, **data_best, **kwargs)
        if sklearn_model:
            model = model_class(**best)
            X = train_data.get_X()
            y = train_data.get_y()
            model.fit(X, y)
        else:
            model : nn.Module = model_class(num_features*data_best['sequence_length'] if time_series and flatten else num_features, **best).to(device)
            train_loader = DataLoader(train_data, batch_size=best['batch_size'])
            optimizer = torch.optim.SGD(model.parameters(), lr=best['lr'])
            for t in range(best['epochs']):
                    print(f"Epoch {t+1}")
                    train(train_loader, model, loss_fn, optimizer, device)
        models.append(model)


    if skip_eval:
        return models, best | data_best, history
    # visualize the training performance of the final model across time
    dt = datetime.datetime.now()
    plot_predictions(f'images/trainplot-{dt.year}-{dt.month:02}-{dt.day:02}-{dt.hour:02}:{dt.minute:02}:{dt.second:02}::{model_name}-{num_folds}fold.png', models, train_data, best, device)

    return models, best | data_best, history

def plot_predictions(file : str, models : list[object], data : AmeriFLUXDataset, hyperparams : dict, device : str, train=True):
    _, test_idx = data.get_train_test_idx(0)
    # option: look at just the final ~6 months
    test_idx = test_idx[-len(test_idx)//2:]
    test_subsampler = SubsetRandomSampler(test_idx)
    dates = data.get_dates(test_idx)
    x = [d.date() for d in dates]
    y_predictions = []
    y = [data.get_y()[idx] for idx in test_idx]
    for model in models:
        if model.__class__.__name__ in ['XGBoost', 'RandomForest']:
            X = [data.get_X()[idx] for idx in test_idx]
            y_pred = model(X)
            y_predictions.append(y_pred)
        else:
            test_loader = DataLoader(data, batch_size=len(dates), shuffle=False)#, sampler=test_subsampler)
            X, _y = next(iter(test_loader))
            X = X.to(device)
            _y = _y.to(device)
            _y_pred : torch.Tensor = model(X)
            y_pred = [a[0] for a in _y_pred.detach().cpu().numpy()]
            y_predictions.append(y_pred)
            #y = [a[0] for a in _y.detach().cpu().numpy()]
    y_predictions = np.array(y_predictions)
    y_pred_avg = y_predictions.transpose().mean(axis=1)
    y_pred_var = y_predictions.transpose().var(axis=1)

    plt.clf()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x, y, label='Actual NEE')

    if len(models) > 1:
        plt.plot(x, y_pred_avg, label='Average predicted NEE')
        for y_prediction in y_predictions:
            plt.scatter(x, y_prediction, c='b', s=3.0, alpha=0.5)
        plt.fill_between(x, y_pred_avg - y_pred_var, y_pred_avg + y_pred_var, alpha=0.2)
    else:
        plt.plot(x, y_predictions.flatten(), label='Predicted NEE')

    plt.xticks([x[i] for i in range(0, len(x), len(x)//12)])
    plt.gcf().autofmt_xdate()
    plt.ylabel("NEE")
    plt.legend()
    plt.title(f"NEE Model Predictions on final year of {'training' if train else 'evaluation'} data")
    subtitle = f"{type(model).__name__}"
    for hparam, value in hyperparams.items():
        if hparam=='r2':
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
    dt = datetime.datetime.now()
    name = model.__class__.__name__
    plt.savefig(file)

def feature_pruning(model_class : Type[nn.Module], site : Site, **kwargs) -> list[str]:
    input_columns : list[str] = kwargs.get('input_columns', None)
    stat_interval = kwargs.get('stat_interval', False)
    # TODO: we need to somehow be able to have rolling window vars without their parent var (e.g. P_rolling_var without P)
    # if no columns are given, start with the full set of variables
    if input_columns is None:
        input_columns = get_site_vars(site)
    # exclude required columns from the feature pruning process
    required_cols = ['TIMESTAMP_START','NEE_PI','NEE_PI_F','USTAR']
    for required_col in required_cols:
        if required_col in input_columns:
            input_columns.remove(required_col)
    # PPFD_IN is required
    if 'PPFD_IN' not in input_columns:
        input_columns.append('PPFD_IN')
    # TODO: prune until we no longer improve r-squared?
    # or prune a specific # of features
    # or test every combination of columns O(2^n) and pick the best one
    # get initial model performance
    history = {}
    kwargs.pop('input_columns', None)
    _, results, _ = train_hparam(model_class, input_columns=input_columns, eval_years=1, skip_eval=True, **kwargs)
    max_r2 = results['r2']
    # history indexed by # of pruned vars
    history[0] = {'pruned_col': None, 'r2': results['r2']}
    pruned_columns = input_columns
    pruning = True
    cols_pruned = 0
    while pruning:
        new_columns = pruned_columns.copy()
        pruned_column = None
        for pruned_idx in range(len(pruned_columns)):
            # do not prune PPFD_IN
            if pruned_columns[pruned_idx] == 'PPFD_IN':
                continue
            print(f"Training model without {pruned_columns[pruned_idx]}")
            _candidate_columns = pruned_columns[0:pruned_idx] + pruned_columns[pruned_idx+1:]# if stat interval is included then 

            _, results, _ = train_hparam(model_class, input_columns=_candidate_columns, skip_eval=True, **kwargs)
            _r2 = results['r2']
            if _r2 > max_r2:
                new_columns = _candidate_columns
                pruned_column = pruned_columns[pruned_idx]
                max_r2 = _r2
        # if we did not find a better set of columns, then end the pruning process
        if len(new_columns) == len(pruned_columns):
            pruning=False
        else:
            cols_pruned += 1
            pruned_columns = new_columns
            history[cols_pruned] = {'pruned_col': pruned_column, 'r2': max_r2}

    # once pruning is complete, write the history to a file
    dt = datetime.datetime.now()
    with open(f"results/pruning_results-{dt.year}-{dt.month:02}-{dt.day:02}::{dt.hour:02}:{dt.minute:02}:{dt.second:02}.md", 'w') as f:
        f.write('### Feature Pruning History\n\n')
        f.write('| # Pruned Columns | Last Pruned Column | Average R-Squared |\n')
        f.write('| --- | --- | --- |\n')
        for num_pruned in range(len(history.keys())):
            f.write(f'| {num_pruned} | {history[num_pruned]["pruned_col"]} | {history[num_pruned]["r2"]:.4f} |\n')

        f.write('\n')
        f.write('## Best Performing feature set:\n\n')
        for col in pruned_columns:
            f.write(f'{col}\n\n')
    return pruned_columns
    

def train_test_eval(model_class : Type[nn.Module], **kwargs) -> float:
    num_folds = kwargs.get('num_folds', 5)
    site = kwargs.get('site', Site.Me2)
    input_columns=kwargs.get('input_columns', [])
    time_series = kwargs.get('time_series', False)
    skip_eval = kwargs.get('skip_eval', False)
    model_name = model_class.__name__
    sklearn_model = model_name in ['XGBoost', 'RandomForest']

    # Perform grid search with k-fold cross validation to optimize the hyperparameters
    final_models, hparams, history = train_hparam(model_class, **kwargs)
    if len(final_models) == 0:
        print("train_hparam failed to train any models")
        return -np.inf

    # Evaluate the final models on the evaluation set
    data_kwargs = kwargs.copy()
    data_kwargs.update(hparams)
    if time_series:
        kwargs.pop('sequence_length', None)
        _, eval_data = prepare_data(site, **data_kwargs)
    else:
        kwargs.pop('stat_interval', None)
        _, eval_data = prepare_data(site, **data_kwargs)

    r2evals = []
    y = eval_data.get_y()
    for model in final_models:
        if sklearn_model:
            y_pred = model(eval_data.get_X())
            r2evals.append(r2_score(y, y_pred))
            device = None
        else:
            eval_loader = DataLoader(eval_data, batch_size=64)
            device = ("cuda" if torch.cuda.is_available() else "cpu")
            r2metric = R2Score(device=device)
            r2evals.append(eval(eval_loader, model, r2metric, device))

    if skip_eval:
        return np.mean(r2evals)

    dt = datetime.datetime.now()
    # plot evaluation performance
    plot_predictions(f'images/evalplot-{dt.year}-{dt.month:02}-{dt.day:02}-{dt.hour:02}:{dt.minute:02}:{dt.second:02}::{model_name}-{num_folds}fold.png', final_models, eval_data, hparams, device, train=False)

    # write the results to an output file
    with open(f"results/training_results-{dt.year}-{dt.month:02}-{dt.day:02}-{dt.hour:02}:{dt.minute:02}:{dt.second:02}::{model_name}-{num_folds}fold.md", 'w') as f:
        f.write("### K-Fold Cross-Validation History\n\n")
        f.write(f'## {model_name}\n\n')
        f.write(f"Folds: {num_folds}\n\n")

        table_head = '|'
        head_border = '|'
        for hparam in hparams.keys():
            if hparam=='r2':
                continue
            title = HYPERPARAMETER_NAMES[hparam]['title']
            table_head += f' {title} |'
            head_border += ' --- |'
        #table_head += ' Average R-Squared |'
        #head_border += ' --- |'

        f.write(f'{table_head} Training Set Size | Average R-Squared |\n')
        f.write(f'{head_border} --- | --- |\n')
        for candidate in history:
            row = '|'
            for hparam in hparams.keys():
                if hparam=='r2':
                    continue
                value = candidate[hparam]
                if isinstance(value, float):
                    row += f' {value:.2f} |'
                elif isinstance(value, Callable):
                    row += f' {value.__name__} |'
                else:
                    row += f' {value} |'
            row += f' {candidate["train_size"]} | {candidate["r2"]:.3f} |'
            f.write(f"{row}\n")
            
        f.write('\n')
        f.write("### Best Model Evaluation\n\n")
        model_str_lines = str(final_models[0]).splitlines()
        f.write('~~~\n')
        for line in model_str_lines[:-1]:
            f.write(f'{line}\n\n')
        f.write(f'{model_str_lines[-1]}\n~~~')
        f.write('\n\n')

        f.write(f'{table_head} Evaluation R-Squared |\n')
        f.write(f'{head_border} --- |\n')

        row = '|'
        for hparam in hparams.keys():
            if hparam=='r2':
                continue
            value = hparams[hparam]
            if isinstance(value, float):
                row += f' {value:.2f} |'
            elif isinstance(value, Callable):
                row += f' {value.__name__} |'
            else:
                row += f' {value} |'
        f.write(f"{row} {np.mean(r2evals):.3f} |\n")
    
    return np.mean(r2evals)