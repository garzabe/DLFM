import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score
from torch import nn
from typing import Callable, Type
import datetime

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
        print(f"Fold {fold}: ")
        # set up next model
        model : nn.Module = model_class(num_features, **model_kwargs).to(device)
        if fold==0:
            print(model)
        # start with the last year as the first validation year and work our way back with each fold
        train_idx, test_idx = train_data.get_train_test_idx(fold)

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

# 
def train_hparam(model_class : Type[nn.Module], **kwargs) -> nn.Module:
    # Accepted kwargs
    epochs = kwargs.get('epochs', 100)
    num_folds = kwargs.get('num_folds', 5)
    input_columns = kwargs.get('input_columns', [])
    site = kwargs.get('site', Site.Me2)
    lr_candidates = kwargs.get('lr_candidates', [1e-1, 1e-2, 1e-4, 1e-5])
    batch_size_candidates = kwargs.get('batch_size_candidates', [1, 4, 16, 32, 64, 128])
    input_columns = kwargs.get('input_columns', None)
    stat_interval =  kwargs.get('stat_interval', None)
    layer_dims = kwargs.get('layer_dims', (4,6))
    activation_fn = kwargs.get('activation_fn', nn.ReLU)

    num_features = len(input_columns)*(3 if stat_interval is not None else 1)

    # 1. Prepare the data
    train_data, _ = prepare_data(site, **kwargs)
    #train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 2. Initialize the model
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    # 2.5 perform hyperparameter tuning to determine best learning rate and batch size

    # let's just be greedy and tune each one independently
    lr_best = lr_candidates[0]
    max_r2 = 0
    for lr in lr_candidates:
        r2 = train_kfold(num_folds, model_class, lr, 64, epochs, loss_fn, train_data, device, num_features, layer_dims=layer_dims, activation_fn=activation_fn)
        if r2 > max_r2:
            lr_best = lr
            max_r2 = r2
    
    max_r2 = 0
    bs_best = batch_size_candidates[0]
    for bs in batch_size_candidates:
        r2 = train_kfold(num_folds, model_class, lr_best, bs, epochs, loss_fn, train_data, device, num_features, layer_dims=layer_dims, activation_fn=activation_fn)
        if r2 > max_r2:
            max_r2 = r2
            bs_best = bs

    # 3. Train with final hparam selections
    model : nn.Module = model_class(num_features, layer_dims=layer_dims, activation_fn=activation_fn).to(device)
    train_loader = DataLoader(train_data, batch_size=bs_best)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_best)
    for t in range(epochs):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer, device)
    return model


def train_test_eval(model_class : Type[nn.Module], **kwargs) -> float:
    num_folds = kwargs.get('num_folds', 5)
    epochs = kwargs.get('epochs', 100)
    site = kwargs.get('site', Site.Me2)
    input_columns = kwargs.get('input_columns', None)
    stat_interval = kwargs.get('stat_interval', None)
    layer_dims = kwargs.get('layer_dims', (4,6))
    activation_fn = kwargs.get('activation_fn', nn.ReLU)

    # Perform grid search with k-fold cross validation to optimize the hyperparameters
    final_model = train_hparam(model_class, **kwargs)

    # Evaluate the final model on the evaluation set
    _, eval_data = prepare_data(site, **kwargs)
    eval_loader = DataLoader(eval_data, batch_size=64)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    r2metric = R2Score(device=device)
    r2eval =  eval(eval_loader, final_model, r2metric, device)

    # write the results to an output file
    dt = datetime.datetime.now()
    with open(f"results/training_results-{dt.year}-{dt.month:02}-{dt.day:02}::{dt.hour:02}:{dt.minute:02}:{dt.second:02}.txt", 'w') as f:
        f.write(f'Layer Architecture: {layer_dims}\n')
        f.write(f'Activation: {activation_fn}\n')
        f.write(f'Folds: {num_folds}\n')
        f.write(f'Epochs: {epochs}\n\n')
        f.write(f'{final_model}')
        f.write('\n\n')
        f.write(f'Evaluation r-squared: {r2eval}\n')
    
    return r2eval