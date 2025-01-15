import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score
from torch import nn
from typing import Callable, Type
import datetime
import numpy as np
import itertools

from data_handler import AmeriFLUXDataset, prepare_data, Site, get_site_vars


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

    # use our own fold indexing
    for fold in range(num_folds):
        print(f"Fold {fold+1}: ")
        # set up next model
        model : nn.Module = model_class(num_features, batch_size=bs, **model_kwargs).to(device)
        if fold==0:
            print(model)
        # start with the last year as the first validation year and work our way back with each fold
        train_idx, test_idx = train_data.get_train_test_idx(fold)
        if train_idx is None:
            print("No more one-year folds can be done, ending K-Fold cross validation")
            break

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
        print("Completed optimization")

        # compute the r squared value for this model
        r2metric = R2Score(device=device)
        r2_results[fold] = eval(test_loader, model, r2metric, device)

    # display r squared results
    print("K-Fold Cross Validation Results")
    sum = 0
    for key, value in r2_results.items():
        print(f"Fold {key}: r2={value:.4f}")
        sum += value
    avg_r2 = sum/len(r2_results.items())
    print(f"Average r-squared: {avg_r2}")
    return avg_r2



def train_hparam(model_class : Type[nn.Module], **kwargs) -> nn.Module:
    # Accepted kwargs
    
    num_folds = kwargs.get('num_folds', 5)
    input_columns = kwargs.get('input_columns', [])
    site = kwargs.get('site', Site.Me2)
    
    # Optimizable hyperparameters - provide a list of test values if we want to tune, otherwise provide a single value
    epochs = kwargs.get('epochs', 100)
    layer_dims = kwargs.get('layer_dims', (4,6))
    activation_fn = kwargs.get('activation_fn', nn.ReLU)
    lr = kwargs.get('lr', 1e-2)
    batch_size = kwargs.get('batch_size', 64)
    stat_interval =  kwargs.pop('stat_interval', None)

    # 2. Initialize the model
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    # Construct the test combinations to test
    epochs_list = epochs if isinstance(epochs, list) else [epochs]
    layer_dims_list = layer_dims if isinstance(layer_dims, list) else [layer_dims]
    activation_fn_list = activation_fn if isinstance(activation_fn, list) else [activation_fn]
    lr_list = lr if isinstance(lr, list) else [lr]
    batch_size_list = batch_size if isinstance(batch_size, list) else [batch_size]
    stat_interval_list = stat_interval if isinstance(stat_interval, list) else [stat_interval]

    data_candidates = [{'stat_interval': si} for si in stat_interval_list]
    candidates = []
    for e, ld, af, l, bs, in itertools.product(epochs_list, layer_dims_list, activation_fn_list, lr_list, batch_size_list):
        candidate = {"epochs": e, "layer_dims": ld, "activation_fn": af, "lr": l, "batch_size": bs}
        candidates.append(candidate)

    print(f"Running K-Fold Cross Validation on {len(candidates)*len(data_candidates)} different hyperparameter configurations")
    history = []
    best = None
    data_best = None
    max_r2 = -np.inf
    for data_candidate in data_candidates:
        train_data, _ = prepare_data(site, stat_interval=data_candidate['stat_interval'], **kwargs)
        if train_data == None or len(train_data) == 0:
            print("Training set had no data. Skipping this candidate...")
            continue
        num_features = len(input_columns)*(3 if data_candidate['stat_interval'] is not None else 1)
        for candidate in candidates:
            print(f"""Hyperparameters:
Stat Interval: {data_candidate['stat_interval']}
Learning Rate: {candidate['lr']}
Batch Size: {candidate['batch_size']}
Epochs: {candidate['epochs']}
Layer Dimensions: {candidate['layer_dims']}
Activation Function: {candidate['activation_fn'].__name__}
""")
            r2 = train_kfold(min(num_folds, train_data.get_num_years()), model_class,
                            candidate['lr'],
                            candidate['batch_size'],
                            candidate['epochs'],
                            loss_fn, train_data, device, num_features,
                            layer_dims=candidate['layer_dims'],
                            activation_fn=candidate['activation_fn'])
            history.append(candidate | data_candidate | {'train_size' : len(train_data)})
            history[-1]['r2'] = r2
            if r2 > max_r2:
                best = candidate
                data_best = data_candidate
                max_r2 = r2
    if max_r2 == -np.inf:
        print("No candidates succeeded (likely all datasets were empty)")
        return None, {'r2': -np.inf}, None
    best['r2'] = max_r2

    

    # 3. Train with final hparam selections
    print("Training the best performing model on the entire training set")
    train_data, _ = prepare_data(site, stat_interval=data_best['stat_interval'], **kwargs)
    num_features = len(input_columns)*(3 if data_best['stat_interval'] is not None else 1)
    model : nn.Module = model_class(num_features, layer_dims=best['layer_dims'], activation_fn=best['activation_fn'], batch_size=best['batch_size']).to(device)
    train_loader = DataLoader(train_data, batch_size=best['batch_size'])
    optimizer = torch.optim.SGD(model.parameters(), lr=best['lr'])
    for t in range(best['epochs']):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer, device)
    return model, best | data_best, history

def feature_pruning(model_class : Type[nn.Module], site : Site, **kwargs) -> list[str]:
    input_columns : list[str] = kwargs.get('input_columns', None)
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
    _, results, _ = train_hparam(model_class, input_columns=input_columns, eval_years=1, **kwargs)
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
            _candidate_columns = pruned_columns[0:pruned_idx] + pruned_columns[pruned_idx+1:]
            _, results, _ = train_hparam(model_class, input_columns=_candidate_columns, **kwargs)
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

    # Perform grid search with k-fold cross validation to optimize the hyperparameters
    final_model, hparams, history = train_hparam(model_class, **kwargs)
    if final_model == None:
        print("train_hparam failed to train any models")
        return -np.inf

    # Evaluate the final model on the evaluation set
    kwargs.pop('stat_interval', None)
    _, eval_data = prepare_data(site, stat_interval=hparams['stat_interval'], **kwargs)
    eval_loader = DataLoader(eval_data, batch_size=64)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    r2metric = R2Score(device=device)
    r2eval =  eval(eval_loader, final_model, r2metric, device)

    # write the results to an output file
    dt = datetime.datetime.now()
    with open(f"results/training_results-{dt.year}-{dt.month:02}-{dt.day:02}::{dt.hour:02}:{dt.minute:02}:{dt.second:02}.md", 'w') as f:
        f.write("### K-Fold Cross-Validation History\n\n")
        f.write(f'## {final_model.__class__.__name__}')
        f.write(f"Folds: {num_folds}\n\n")

        f.write('| Training Set Size | Layer Dimensions | Activation Function | Learning Rate | Batch Size | Epochs | Time Series Interval (days) | Average R-Squared |\n')
        f.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for candidate in history:
            f.write(f"| {candidate['train_size']} | {candidate['layer_dims']} | {candidate['activation_fn'].__name__} | {candidate['lr']} | {candidate['batch_size']} | {candidate['epochs']} | {candidate['stat_interval']} | {candidate['r2']:.4f} |\n")

        f.write('\n')
        f.write("### Best Model Evaluation\n\n")
        model_str_lines = str(final_model).splitlines()
        f.write('~~~\n')
        for line in model_str_lines[:-1]:
            f.write(f'{line}\n\n')
        f.write(f'{model_str_lines[-1]}\n~~~')
        f.write('\n\n')

        f.write('| Layer Dimensions | Activation Function | Learning Rate | Batch Size | Epochs | Time Series Interval (days) | Evaluation R-Squared |\n')
        f.write('| --- | --- | --- | --- | --- | --- | --- |\n')
        f.write(f"| {hparams['layer_dims']} | {hparams['activation_fn'].__name__} | {hparams['lr']} | {hparams['batch_size']} | {hparams['epochs']} | {candidate['stat_interval']} | {r2eval:.4f} |\n")
    
    return r2eval