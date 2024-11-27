from enum import Enum
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torcheval.metrics import R2Score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tqdm
from typing import Type, Callable

global input_column_set
global site



Site = Enum('Site', ['Me2', 'Me6'])

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
    
class AmeriFLUXSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = X
        self.labels = y

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]

        return input, label

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

input_column_set = me2_input_column_set

LAYER_FEATURES = 8

# Elman RNN
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_state_size = 8
        global input_column_set
        self.rnn = nn.RNN(len(input_column_set), hidden_state_size, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_state_size, out_features=1)
    
    def forward(self, x):
        _batch_size = x.shape[0]
        h0 = torch.zeros(1, _batch_size, 8)
        # we are only interested in the final output (at the moment)
        _, hn = self.rnn(x, h0)
        _hn = self.relu(hn)
        return self.linear(_hn)[0]

# returns a training dataset and a test dataset with sequences of the given length (half hour intervals)
# for reference, 24 hours == length of 48, 1 week == length 336, 1 month (31 days) == 1488, 1 year == 17520
def prepare_data(site_name : Site, sequence_length : int, no_split = False) -> tuple[AmeriFLUXDataset, AmeriFLUXDataset]:
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
    

    # remove daylight hours?
    #df = df[df['PPFD_IN'] > 4.0]

    _nrows = len(df)

    # reduce the columns to our desired set
    target_col = 'NEE_PI' if 'NEE_PI' in df.columns else 'NEE_PI_F'

    global input_column_set
    df['DATETIME'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df = df[['DATETIME', *input_column_set, target_col]]
    df["NEE"] = df[target_col]
    df = df.drop(columns=[target_col])

    X_dataset = []
    y_dataset = []
    # TODO: create the sequences for time-series predictors
    print("Building dataset")
    t = tqdm.tqdm(total = len(df)-sequence_length)
    for i in range(len(df)-sequence_length):
        t.update(1)
        # if there are any gaps, skip this iteration
        if df.iloc[i:i+sequence_length].isna().values.any():
            continue
        df_seq = df.iloc[i:i+sequence_length].dropna().drop(columns=['DATETIME'])
        #t_0 = df_seq["DATETIME"].iloc[0]
        # get time difference from t_0 in minutes, these will be our new columns?
        #delta_t = (df_seq["DATETIME"] - t_0).apply(lambda td: td.days*48 + td.seconds // 60)
        X_seq = df_seq.drop(columns=["NEE"]).to_numpy(dtype=np.float32)
        #print(X_seq)
        y = df_seq[["NEE"]].iloc[-1].to_numpy(dtype=np.float32)
        X_dataset.append(X_seq)
        y_dataset.append(y)
    print(f"The final size of the dataset is {len(X_dataset)}")
    train_size = int(len(X_dataset)*0.8)
    X_train = np.array(X_dataset[0:train_size])
    y_train = np.array(y_dataset[0:train_size])
    X_eval = np.array(y_dataset[train_size:])
    y_eval = np.array(y_dataset[train_size:])
    return AmeriFLUXSequenceDataset(X_train, y_train), AmeriFLUXSequenceDataset(X_eval, y_eval)



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

        history = []

        for t in range(epochs):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer, device)
            history.append(test(test_loader, model, loss_fn, device))
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

def train_test(model_class : Type[nn.Module], model_args = [], epochs : int = 100, num_folds : int = 5) -> nn.Module:
    global site
    # 1. Prepare the data
    global input_column_set
    input_column_set = me2_input_column_set if site==Site.Me2 else me6_input_column_set
    train_data, _ = prepare_data(site, 48)
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
        r2 = train_kfold(num_folds, model_class, lr, 64, epochs, loss_fn, train_data, device, model_args)
        if r2 > max_r2:
            lr_best = lr
            max_r2 = r2
    
    max_r2 = 0
    bs_best = batch_size_candidates[0]
    for bs in batch_size_candidates:
        r2 = train_kfold(num_folds, model_class, lr_best, bs, epochs, loss_fn, train_data, device, model_args)
        if r2 > max_r2:
            max_r2 = r2
            bs_best = bs

    # 3. Train with final hparam selections
    model : nn.Module = model_class(*model_args).to(device)
    train_loader = DataLoader(train_data, batch_size=bs_best)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_best)
    for t in range(epochs):
            print(f"Epoch {t+1}")
            train(train_loader, model, loss_fn, optimizer, device)
    return model
        

def train_test_eval(model_class : Type[nn.Module], model_args = [], num_folds : int = 5, epochs : int = 100) -> float:
    global site
    final_model = train_test(model_class, model_args=model_args, num_folds=num_folds, epochs=epochs)
    _, eval_data = prepare_data(site, 48)
    eval_loader = DataLoader(eval_data, batch_size=64)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    r2metric = R2Score(device=device)
    return eval(eval_loader, final_model, r2metric, device)



def main():
    print(train_test_eval(RNN))


if __name__=="__main__":
    main()