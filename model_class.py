import torch
from torch import nn
from typing import Callable
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# This dictionary has a lot of use-cases:
# train_hparams queries this to determine which kwargs it should look for, and what values to default to for each model
# plot_predictions uses this to determine what hyperparameters to write in the subtitle
# train_test_eval uses this similarly, to determine how to build the history table

MODEL_HYPERPARAMETERS : dict[str, dict[str, int | tuple[int] | Callable | None]] = {
    'XGBoost' : {'lr' : 0.01, 'n_estimators' : 1000, 'sequence_length': None, 'stat_interval': None},
    'RandomForest' : {'n_estimators' : 100, 'sequence_length': None, 'stat_interval': None},
    'FirstANN' : {'epochs' : 300, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.001, 'stat_interval' : None, 'sequence_length' : None, 'weight_decay': 0.0, 'momentum': 0.0},
    'DynamicANN' : {'layer_dims' : (6,6), 'epochs' : 300, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.001, 'stat_interval' : None, 'sequence_length' : None, 'weight_decay': 0.0, 'momentum': 0.0},
    'RNN' : {'hidden_state_size' : 8, 'num_layers' : 1, 'epochs' : 800, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.001, 'sequence_length' : 14, 'dropout' : 0.0, 'weight_decay': 0.0, 'momentum': 0.0},
    'LSTM' : {'hidden_state_size' : 8, 'num_layers' : 1, 'epochs' : 800, 'batch_size' : 64, 'activation_fn' : nn.ReLU, 'lr' : 0.001, 'sequence_length' : 14, 'dropout' : 0.0, 'weight_decay': 0.0, 'momentum': 0.0},
    'xLSTM' : {'epochs' : 500, 'batch_size' : 64, 'lr' : 0.001, 'sequence_length' : 14, 'dropout' : 0.0, 'weight_decay': 0.0, 'momentum': 0.0}
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

    def forward(self, x : torch.Tensor):
        out =  self.stack(x)
        return out

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

        self.h0 = torch.zeros(hidden_state_size, requires_grad=True).to(("cuda" if torch.cuda.is_available() else "cpu"))
        self.c0 = torch.zeros(hidden_state_size, requires_grad=True).to(("cuda" if torch.cuda.is_available() else "cpu"))

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

    # defines the output of the upsampling layer and the input size to the sLSTM
    # D in the paper
    LATENT_DIM = 20
    # size of the sLSTM stack - M in the paper
    sLSTM_SIZE = 2

    def __init__(self, num_features : int, sequence_length : int = -1, hidden_state_size=8, dropout=0, **kwargs):

        super().__init__()

        if sequence_length == -1:
            raise ValueError("Please provide a sequence length for the xLSTM")
        
        self.num_features = num_features
        self.sequence_length = sequence_length

        ### Time Mixing
        # xLSTM first processes each variable independently through a fully connected (linear) layer
        # NLinear shares weights across all variables
        self.NLinear = nn.Linear(in_features=sequence_length, out_features=sequence_length)


        ### Joint Mixing
        # upsampling layer with weight-sharing
        self.FC_up = nn.Linear(in_features=sequence_length, out_features=xLSTM.LATENT_DIM)

        # each time-series gets prepended with an initial learnable embedding
        self.mu = torch.randn(xLSTM.LATENT_DIM).to(("cuda" if torch.cuda.is_available() else "cpu"))

        # transpose the (concatenated) input
        self.transpose = lambda t: torch.transpose(t, 1, 2)

        self.reverse = lambda t: torch.flip(t, (1, ))

        # the main component: sLSTM stack
        # from the github implementation, it seems there is no up/downsampling done in the sLSTM
        # so the hidden state is the same size as the input
        self.h0 = torch.zeros(xLSTM.LATENT_DIM).to(("cuda" if torch.cuda.is_available() else "cpu"))
        self.c0 = torch.zeros(xLSTM.LATENT_DIM).to(("cuda" if torch.cuda.is_available() else "cpu"))
        self.sLSTM = nn.LSTM(xLSTM.LATENT_DIM, xLSTM.LATENT_DIM, num_layers=xLSTM.sLSTM_SIZE, dropout=dropout, batch_first=True)

        ### View Mixing
        # inputs are re-transposed before input to FC_view
        # TODO: uncertain about the input shape here
        self.FC_view = nn.Linear(in_features=2*xLSTM.LATENT_DIM, out_features=1)

    def forward(self, x):
        # x has shape (batch size, sequence length, num features)
        # with drop_last=True, the batch size should be consistent
        _batch_size = x.shape[0]

        #print(f"Before Time Mixer, x has shape {x.shape}")

        x_up = torch.zeros(_batch_size, xLSTM.LATENT_DIM, self.num_features).to(("cuda" if torch.cuda.is_available() else "cpu"))

        # separate each feature and pass through the Time Mixer component
        for i in range(len(x[0][0])):
            _x = x[:, :, i]
            #print(f"The individual variable has shape {_x.shape}")
            _x = self.NLinear(_x)
            _x = self.FC_up(_x)
            x_up[:, :, i] = _x

        #print(f"After Time Mixer, x_up has shape {x_up.shape}")
        _mu = self.mu.repeat(_batch_size, 1, 1)
        #print(_mu.shape)

        # Transpose to enter the Joint Mixer component
        x_t = self.transpose(x_up)
        x_t_r = self.reverse(x_t)

        #print(f"After transposition, before Joint Mixer, x_t has shape {x_t.shape}")
        #print(f"After transposition, before Joint Mixer, x_t_r has shape {x_t_r.shape}")

        # prepend mu to each input
        x_t_primed = torch.concat([_mu, x_t], dim=1)
        x_t_r_primed = torch.concat([_mu, x_t_r], dim=1)
        #print(f"After prepending, x_t_primed has shape {x_t_primed.shape}")


        _h0 = self.h0.repeat(xLSTM.sLSTM_SIZE, _batch_size, 1)
        _c0 = self.c0.repeat(xLSTM.sLSTM_SIZE, _batch_size, 1)

        #print(f"The initial hidden state and cell state have size {_h0.shape}")

        # _, (_x, _) = self.lstm(x, (_h0, _c0))

        # LSTM returns <all hidden state outputs H>, (<final hidden state hT>, <final cell state cT>)
        _y_forward, (_, _) = self.sLSTM(x_t_primed, (_h0, _c0))
        _y_reverse, (_, _) = self.sLSTM(x_t_r_primed, (_h0, _c0))

        #print(f"After pass through sLSTM, both y have shape {_y_forward.shape}")

        # only interested in the output of the final layer
        #_y_forward_final = _y_forward[:, -1, :]
        #_y_reverse_final = _y_reverse[:, -1, :]

        #print(f"The final outputs have shapes {_y_forward_final.shape}")

        # Re-transpose
        _y_f_t = self.transpose(_y_forward)
        _y_f_r = self.transpose(_y_reverse)

        #print(f"After re-transpose, both y_t have shape {_y_f_t.shape}")

        _y = torch.concat([_y_f_t, _y_f_r], axis=1)

        # Original xLSTM model is designed to forecast each input variable
        # since we are interested in predicting an entirely different variable not in the input,
        # we modify the output process slightly

        # take the final joint mixed tensor, and pass through the FC_view to get a prediction for NEP on final day

        _y_final = _y[:, :, -1]

        #print(f"The final-var output (retransposed) from the sLSTM has shape {_y_final.shape}")

        y = self.FC_view(_y_final)

        #print(f"The final output has shape {y.shape}")

        return y




            
        
        

        







#-------------------------
class XGBoost():
    def __init__(self, lr=None, n_estimators=None, **kwargs):
        model_kwargs = {'eval_metric': mean_squared_error}#, 'early_stopping_rounds': epochs}
        if lr is not None:
            model_kwargs['learning_rate'] = lr
        if n_estimators is not None:
            model_kwargs['n_estimators'] = n_estimators
        self.model : XGBRegressor = XGBRegressor(**model_kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y, eval_set=[(X, y)], verbose=False)

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
        self.model : RandomForestRegressor = RandomForestRegressor(verbose=0, **model_kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def __call__(self, X):
        return self.model.predict(X)
    
    def __str__(self):
        return str(self.model)