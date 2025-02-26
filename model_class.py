import torch
from torch import nn
from typing import Callable
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# This dictionary has a lot of use-cases:
# train_hparams queries this to determine which kwargs it should look for, and what values to default to for each model
# plot_predictions uses this to determine what hyperparameters to write in the subtitle
# train_test_eval uses this similarly, to determine how to build the history table

MODEL_HYPERPARAMETERS : dict[str, dict[str, int | tuple[int] | Callable | None]] = {
    'XGBoost' : {'lr' : 0.5, 'n_estimators' : 100},
    'RandomForest' : {'n_estimators' : 100},
    'FirstANN' : {'epochs' : 100, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.01, 'stat_interval' : None, 'sequence_length' : None},
    'DynamicANN' : {'layer_dims' : (4,6), 'epochs' : 100, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.01, 'stat_interval' : None, 'sequence_length' : None},
    'RNN' : {'hidden_state_size' : 8, 'num_layers' : 1, 'epochs' : 100, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.01, 'sequence_length' : 14, 'dropout' : 0.0},
    'LSTM' : {'hidden_state_size' : 8, 'num_layers' : 1, 'epochs' : 100, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.01, 'sequence_length' : 14, 'dropout' : 0.0},
    'xLSTM' : {}
}

# each class should have an __init__
class NEPModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        pass


class FirstANN(NEPModel):
    def __init__(self, num_features : int, **kwargs):
        # accepted kwargs: layer_size
        layer_size = kwargs.get('layer_size', 8)

        super().__init__()
        self.stack  = nn.Sequential(
            # 8 hidden layer nodes from the 2017 paper
            nn.Linear(in_features=num_features, out_features=layer_size),
            nn.ReLU(),
            nn.Linear(in_features=layer_size, out_features=1)
        )

    def forward(self, x):
        return self.stack(x)
    
class DynamicANN(NEPModel):
    def __init__(self, num_features : int, **kwargs):
        # Accepted kwargs
        layer_dims = kwargs.get('layer_dims', (4,6))
        activation_fn = kwargs.get('activation_fn', nn.ReLU)
        activation_args = kwargs.get('activation_args', [])
        
        super().__init__()
        sequential_layers = [
            nn.Linear(in_features=num_features, out_features=layer_dims[0]),
            activation_fn(*activation_args),
        ]
        for i in range(1, len(layer_dims)):
            sequential_layers.append(nn.Linear(in_features=layer_dims[i-1], out_features=layer_dims[i]))
            sequential_layers.append(activation_fn(*activation_args))
        sequential_layers.append(nn.Linear(in_features=layer_dims[-1], out_features=1))
        self.stack  = nn.Sequential(*sequential_layers)

    def forward(self, x):
        return self.stack(x)

class RNN(NEPModel):
    def __init__(self, num_features : int, **kwargs):
        super().__init__()
        hidden_state_size = kwargs.get('hidden_state_size', 8)
        batch_size = kwargs.get("batch_size", 1)
        num_layers = kwargs.get('num_layers', 1)
        self.num_layers = num_layers

        self.h0 = torch.zeros(hidden_state_size).to(("cuda" if torch.cuda.is_available() else "cpu"))

        self.rnn = nn.RNN(num_features, hidden_state_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_state_size, out_features=1)
    
    def forward(self, x):
        # batch size is not necessarily batch_size (i.e. the last batch is less than that)
        _batch_size = x.shape[0]
        _h0 = self.h0.repeat(self.num_layers, _batch_size, 1)
        # we are only interested in the final output (at the moment)
        _, hn = self.rnn(x, _h0)
        _hn = self.relu(hn)
        return self.linear(_hn)[0]
    

class LSTM(NEPModel):
    def __init__(self, num_features : int, **kwargs):
        super().__init__()

        hidden_state_size = kwargs.get('hidden_state_size', 8)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.0)
        batch_size = kwargs.get("batch_size", 1)

        self.num_layers = num_layers

        self.h0 = torch.zeros(hidden_state_size).to(("cuda" if torch.cuda.is_available() else "cpu"))
        self.c0 = torch.zeros(hidden_state_size).to(("cuda" if torch.cuda.is_available() else "cpu"))

        self.lstm = nn.LSTM(num_features, hidden_state_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_state_size, out_features=1)
    
    def forward(self, x):
        _batch_size = x.shape[0]
        _h0 = self.h0.repeat(self.num_layers, _batch_size, 1)
        _c0 = self.c0.repeat(self.num_layers, _batch_size, 1)
        _, (_x, _) = self.lstm(x, (_h0, _c0))
        _x = self.relu(_x)
        y = self.linear(_x)
        return y[0]

# TODO: xLSTMMixer
class xLSTM(NEPModel):

    def __init__(self):
        pass


#-------------------------
class XGBoost():
    def __init__(self, lr=None, n_estimators=None, **kwargs):
        # TODO: manually process model kwargs
        model_kwargs = {}
        if lr is not None:
            model_kwargs['learning_rate'] = lr
        if n_estimators is not None:
            model_kwargs['n_estimators'] = n_estimators
        self.model : XGBRegressor = XGBRegressor(**model_kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    # allows the model to be used as a callable simiar to nn.Module
    def __call__(self, X):
        return self.model.predict(X)
    
    def __str__(self):
        return str(self.model)
    
class RandomForest():
    def __init__(self, n_estimators=None, **kwargs):
        model_kwargs = {}
        if n_estimators is not None:
            model_kwargs['n_estimators'] = n_estimators
        # TODO: process kwargs for randomforest relevant kwargs
        self.model : RandomForestRegressor = RandomForestRegressor(**model_kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def __call__(self, X):
        return self.model.predict(X)
    
    def __str__(self):
        return str(self.model)