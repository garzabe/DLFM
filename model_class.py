import torch
from torch import nn
from typing import Callable
from abc import ABC, abstractmethod

# each class should have an __init__
class NEPModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        pass

# TODO: implement Random Forest, XGBoost architectures
# these don't work in the pytorch framework

# TODO: xLSTM, standard LSTM, 1d CNN

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

        print("Initializing module")

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
    def __init__(self):
        super().__init__()
        hidden_state_size = 8
        global input_column_set
        self.rnn = nn.RNN(len(input_column_set), hidden_state_size, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_state_size, out_features=1)
    
    def forward(self, x):
        _batch_size = x.shape[0]
        h0 = torch.zeros(1, _batch_size, 8).to(("cuda" if torch.cuda.is_available() else "cpu"))
        # we are only interested in the final output (at the moment)
        _, hn = self.rnn(x, h0)
        _hn = self.relu(hn)
        return self.linear(_hn)[0]