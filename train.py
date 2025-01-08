import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score
from torch import nn
from typing import Callable, Type
import datetime
import numpy as np
import itertools

from data_handler import AmeriFLUXDataset, prepare_data, Site


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

    num_features = len(input_columns)*(3 if stat_interval is not None else 1)

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
    best = None
    data_best = None
    max_r2 = -np.inf
    for data_candidate in data_candidates:
        train_data, _ = prepare_data(site, stat_interval=data_candidate['stat_interval'], **kwargs)
        for candidate in candidates:
            print(f"""Hyperparameters:
Stat Interval: {data_candidate['stat_interval']}
Learning Rate: {candidate['lr']}
Batch Size: {candidate['batch_size']}
Epochs: {candidate['epochs']}
Layer Dimensions: {candidate['layer_dims']}
Activation Function: {candidate['activation_fn'].__name__}
""")
            r2 = train_kfold(num_folds, model_class,
                            candidate['lr'],
                            candidate['batch_size'],
                            candidate['epochs'],
                            loss_fn, train_data, device, num_features,
                            layer_dims=candidate['layer_dims'],
                            activation_fn=candidate['activation_fn'])
            if r2 > max_r2:
                best = candidate
                data_best = data_candidate
                max_r2 = r2

    # 3. Train with final hparam selections
    train_data, _ = prepare_data(site, stat_interval=data_best['stat_interval'], **kwargs)
    model : nn.Module = model_class(num_features, layer_dims=best['layer_dims'], activation_fn=best['activation_fn'], batch_size=best['batch_size']).to(device)
    train_loader = DataLoader(train_data, batch_size=best['batch_size'])
    optimizer = torch.optim.SGD(model.parameters(), lr=best['lr'])
    for t in range(best['epochs']):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer, device)
    return model, best | data_best

def train_test_eval(model_class : Type[nn.Module], **kwargs) -> float:
    num_folds = kwargs.get('num_folds', 5)
    site = kwargs.get('site', Site.Me2)
    input_columns=kwargs.get('input_columns', [])

    # Perform grid search with k-fold cross validation to optimize the hyperparameters
    final_model, hparams = train_hparam(model_class, **kwargs)

    # Evaluate the final model on the evaluation set
    kwargs.pop('stat_interval', None)
    _, eval_data = prepare_data(site, stat_interval=hparams['stat_interval'], **kwargs)
    eval_loader = DataLoader(eval_data, batch_size=64)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    r2metric = R2Score(device=device)
    r2eval =  eval(eval_loader, final_model, r2metric, device)

    # write the results to an output file
    dt = datetime.datetime.now()
    with open(f"results/training_results-{dt.year}-{dt.month:02}-{dt.day:02}::{dt.hour:02}:{dt.minute:02}:{dt.second:02}.txt", 'w') as f:
        f.write(f'Layer Architecture: {hparams["layer_dims"]}\n')
        f.write(f'Activation: {hparams["activation_fn"].__name__}\n')
        f.write(f'Learning Rate: {hparams["lr"]}\n')
        f.write(f'Batch Size: {hparams["batch_size"]}\n')
        f.write(f'Folds: {num_folds}\n')
        f.write(f'Epochs: {hparams["epochs"]}\n\n')
        f.write(f'{final_model}')
        f.write('\n\n')
        f.write(f'Evaluation r-squared: {r2eval}\n')
    
    return r2eval