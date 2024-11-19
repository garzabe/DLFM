import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from torch import nn
from torcheval.metrics import R2Score
from enum import Enum
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold
from typing import Type, Callable

global input_column_set
global site

Site = Enum('Site', ['Me2', 'Me6'])

site = Site.Me2

# this is the set for Me-2
me2_input_column_set = [
    'D_SNOW',
    # no data until 2006
    'SWC_1_7_1',
    # 2 7 1 has really spotty data
    #'SWC_2_7_1',
    #'SWC_3_7_1',
    'SWC_1_2_1',
    'RH',
    'NETRAD',
    'PPFD_IN',
    'TS_1_3_1',
    #'V_SIGMA',
    'P',
    'WD',
    'WS',
    # TA 1 1 1 has no data until 2007
    'TA_1_1_3',
]

me6_input_column_set = [
    'D_SNOW',
    'SWC_1_5_1',
    'SWC_1_2_1',
    'RH',
    'NETRAD',
    'PPFD_IN',
    'TS_1_5_1',
    'P',
    'WD',
    'WS',
    'TA_1_1_2'
]

LAYER_FEATURES = 8

class FirstANN(nn.Module):
    def __init__(self):
        super().__init__()
        global input_column_set
        self.stack  = nn.Sequential(
            # 8 hidden layer nodes from the 2017 paper
            nn.Linear(in_features=len(input_column_set), out_features=LAYER_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=LAYER_FEATURES, out_features=1)
        )

    def forward(self, x):
        return self.stack(x)
    
class DynamicANN(nn.Module):
    def __init__(self, layer_dims : list[int], activation_fn : Callable, activation_args = []):
        super().__init__()
        global input_column_set
        sequential_layers = [
            nn.Linear(in_features=len(input_column_set), out_features=layer_dims[0]),
            activation_fn(*activation_args),
        ]
        for i in range(1, len(layer_dims)):
            sequential_layers.append(nn.Linear(in_features=layer_dims[i-1], out_features=layer_dims[i]))
            sequential_layers.append(activation_fn(*activation_args))
        sequential_layers.append(nn.Linear(in_features=layer_dims[-1], out_features=1))
        self.stack  = nn.Sequential(*sequential_layers)

    def forward(self, x):
        return self.stack(x)

class AmeriFLUXDataset(Dataset):
    def __init__(self, df_X_y : pd.DataFrame):
        # hold onto the original dataframe
        self.df = df_X_y
        _df = df_X_y.reset_index()
        self.inputs : pd.DataFrame = _df.drop(columns=['DAY', 'NEE'])
        self.labels : pd.Series = _df['NEE']

    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, idx):
        input : np.ndarray = self.inputs.iloc[idx].drop("index").to_numpy(dtype=np.float32)
        label : np.ndarray = np.array([self.labels.iloc[idx]], dtype=np.float32)
        return input, label


# returns a training dataset and a test dataset
def prepare_data(site_name : Site, no_split = False) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
    filepath = ''
    if site_name == Site.Me2:
        filepath = 'AmeriFLUX Data/AMF_US-Me2_BASE-BADM_19-5/AMF_US-Me2_BASE_HH_19-5.csv'
    elif site_name == Site.Me6:
        filepath = 'AmeriFLUX Data/AMF_US-Me6_BASE-BADM_16-5/AMF_US-Me6_BASE_HH_16-5.csv'
    else:
        print(f"Error: {site_name} not a valid site")

    df = pd.read_csv(filepath, header=2)

    # replace -9999 with NaN
    df = df.replace(-9999, np.nan)

    _nrows = len(df)
    # drop all rows where ustar is not sufficient
    df = df[df['USTAR'] > 0.2]
    #print(f"Dropped {_nrows - len(df)}  rows with USTAR threshold ({len(df)})")
    

    # TODO: improve daylight row selection
    df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    global input_column_set
    df = df[['TIMESTAMP_START', *input_column_set, target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    # group into daily averages
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    df['DAY'] = df['DATETIME'].apply(lambda dt: f"{dt.year:04}{dt.month:02}{dt.day:02}")
    df = df.drop(columns=['DATETIME', 'TIMESTAMP_START'])
    
    # the means are the important values, but count helps us identify low-data days
    df_avg = df.groupby('DAY').aggregate('mean').reset_index()
    df_count = df.groupby('DAY').aggregate('count').reset_index()

    # now only include the days where all column counts are above 20??
    # perfect recording is 48 per day
    # with ~9 hours of daylight, the max daylight rows is 18
    _nrows = len(df_avg)
    min_count = 5
    #print(df_count.head())
    min_count_filter = df_count.drop(columns=['DAY']) >= min_count
    #print(df_avg[~min_count_filter])
    df_X_y = df_avg[min_count_filter.all(axis=1)]
    #print(df_X_y.head())
    #df_X_y = df_X_y.drop(columns=['DAY'])
    print(f"Dropped {_nrows - len(df_X_y)} rows from min count filter ({len(df_X_y)})")

    # TODO: normalize all data
    # add these columns back at the end
    _df = df_X_y.drop(columns=["DAY", "NEE"])
    _df = (_df - _df.mean())/_df.std()
    _df["DAY"] = df_X_y["DAY"]
    _df["NEE"] = df_X_y["NEE"]

    # simple train-test and eval 80/20 split
    _df_eval = _df.iloc[int(len(_df)*0.2):]
    _df = _df.iloc[0:int(len(_df)*0.2)]

    ## TODO: remove this, we are no longer using default train test splitting
    # split in to train, test, validation
    #if not no_split:
    #    df_train, df_test = train_test_split(_df, test_size=0.2)
    #    print(f"The training set has {len(df_train)} entries, and the test set has {len(df_test)} entries")
    #else:
    #    df_train = _df
    #    df_test = None
    #df_train = _df
    #return AmeriFLUXDataset(df_train), AmeriFLUXDataset(df_test) if df_test is not None else None
    return AmeriFLUXDataset(_df), AmeriFLUXDataset(_df_eval)
    

# TODO: have this function return the training history
def train(dataloader : DataLoader,
          model : nn.Module,
          loss_fn : Callable, # not necessarily mseloss
          optimizer : torch.optim.Optimizer) -> None:
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Get predictions and loss value
        pred = model(X)
        loss : torch.Tensor = loss_fn(pred, y)

        # backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # get updates
        #if batch % 4 == 0:
        #    loss, current = loss.item(), batch*64 + len(X)
        #    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def train_kfold(num_folds : int,
                model_class : Type[nn.Module],
                lr : float, bs : int, epochs : int, loss_fn : Callable,
                train_data : AmeriFLUXDataset,
                device,
                model_args = []) -> float:
    
    kfold = KFold(n_splits=num_folds, shuffle=False)

    r2_results = {}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_data)):
        print(f"Fold {fold}: ")
        # set up next model
        model : nn.Module = model_class(*model_args).to(device)
        if fold==0:
            print(model)

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
            train(train_loader, model, loss_fn, optimizer)
            test(test_loader, model, loss_fn)
        print("Completed optimization")

        # compute the r squared value for this model
        r2metric = R2Score(device=device)
        r2_results[fold] = eval(test_loader, model, r2metric)

    # display r squared results
    print("K-Fold Cross Validation Results")
    sum = 0
    for key, value in r2_results.items():
        print(f"Fold {key}: r2={value:.4f}")
        sum += value
    avg_r2 = sum/len(r2_results.items())
    print(f"Average r-squared: {avg_r2}")
    return avg_r2


def test(dataloader : DataLoader, model : nn.Module, loss_fn : nn.MSELoss) -> None:
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f}")

def eval(dataloader : DataLoader, model : nn.Module, metric_fn) -> float:
    size = len(dataloader.dataset)
    model.eval()

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            pred = model(X)
            metric_fn.update(input=pred, target=y)
            
    print(f"Metric value: {metric_fn.compute().item():>8f}")
    return metric_fn.compute().item()

def train_test(model_class : Type[nn.Module], model_args = []) -> nn.Module:
    k_folds = 5
    epochs = 100
    
    global site
    # 1. Prepare the data
    global input_column_set
    input_column_set = me2_input_column_set if site==Site.Me2 else me6_input_column_set
    train_data, _ = prepare_data(site)
    #train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 2. Initialize the model
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()


    # 2.5 perform hyperparameter tuning to determine best learning rate and batch size
    lr_candidates = [1e-1, 1e-2, 1e-4, 1e-5]
    batch_size_candidates = [1, 4, 16, 32, 64, 128]

    # let's just be greedy and tune each one independently
    lr_best = lr_candidates[0]
    max_r2 = 0
    for lr in lr_candidates:
        r2 = train_kfold(k_folds, model_class, lr, 64, epochs, loss_fn, train_data, device, model_args)
        if r2 > max_r2:
            lr_best = lr
            max_r2 = r2
    
    max_r2 = 0
    bs_best = batch_size_candidates[0]
    for bs in batch_size_candidates:
        r2 = train_kfold(k_folds, model_class, lr_best, bs, epochs, loss_fn, train_data, device, model_args)
        if r2 > max_r2:
            max_r2 = r2
            bs_best = bs

    # 3. Train with final hparam selections
    model : nn.Module = model_class(*model_args).to(device)
    train_loader = DataLoader(train_data, batch_size=bs_best)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_best)
    for t in range(epochs):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer)
    return model
        

def train_test_eval(model_class : Type[nn.Module], model_args = []) -> float:
    global site
    final_model = train_test(model_class, model_args=model_args)
    _, eval_data = prepare_data(site)
    eval_loader = DataLoader(eval_data, batch_size=64)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    r2metric = R2Score(device=device)
    return eval(eval_loader, final_model, r2metric)

    """
    # Visualize model performance, and
    # calculate R-Squared value
    r2metric = R2Score(device=device)
    full_data, _ = prepare_data(site, no_split=True)
    full_df = full_data.df
    eval_dataloader = DataLoader(full_data, batch_size=len(full_data), shuffle=False)
    X, _y = next(iter(eval_dataloader))
    pred : torch.Tensor = model(X)
    r2metric.update(input=pred, target=_y)
    print(f"The model achieves an r-squared value of {r2metric.compute().item():.4f}")
    pred_s = [a[0] for a in pred.detach().numpy()]
    y = full_df['NEE'].to_list()
    dates = pd.to_datetime(full_df['DAY'], format="%Y%m%d").to_list()
    x = [d.date() for d in dates]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    limit = 365
    plt.plot(x[:limit], y[:limit], label='Actual NEE')
    plt.plot(x[:limit], pred_s[:limit], label='Predicted NEE')
    plt.xticks([x[i] for i in range(0, limit, limit//10)])
    plt.gcf().autofmt_xdate()
    plt.ylabel("NEE")
    plt.legend()
    plt.title('NEE Model Predictions')
    plt.show()
    """

import datetime
def main():
    # comparing performance of first ann and two layer ann
    first_r2 = train_test_eval(FirstANN)
    layer_sizes = [4,6,8,10,12,14,20]
    results = [('original', first_r2)]
    for l1 in layer_sizes:
        arch = [l1]
        results.append((arch, train_test_eval(DynamicANN, [arch, nn.ReLU])))
    for l1 in layer_sizes:
        for l2 in layer_sizes:
            arch = [l1, l2]
            results.append((arch, train_test(DynamicANN, [arch, nn.ReLU])))

    with open('output.txt', 'a+') as f:
        dt = datetime.datetime.now()
        for arch, r2 in results:
            f.write(f"{dt.year}-{dt.month:02}-{dt.day:02} {dt.hour:02}:{dt.minute:02}:{dt.second:02} : Architecture {arch} R-Squared {r2:.4f}\n")

if __name__=="__main__":
    main()