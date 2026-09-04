import torch
import torch.nn as nn
import enums
from torch_config import resolve_device_dtype



class CCRDeepModel(nn.Module):
    def __init__(self, number_risk_factors, num_rnn_layers, num_rnn_hidden_units, rnn_type, 
                 num_deep_layers, deep_hidden_units, device=None, dtype=None):

        super(CCRDeepModel, self).__init__()

        self.device, self.dtype = resolve_device_dtype(device, dtype)
        self.rnn_type = rnn_type
        factory_kwargs = {"device": self.device, "dtype": self.dtype}
        if num_rnn_layers < 0:
            raise ValueError("num_rnn_layers must be non-negative")
        if num_deep_layers < 0:
            raise ValueError("num_deep_layers must be non-negative")

        # If num_rnn_layers is zero, skip the recurrent block.
        if num_rnn_layers == 0:
            self.rnn = None
            input_size = number_risk_factors
        elif rnn_type == enums.RNNType.RNN:
            self.rnn = nn.RNN(input_size=number_risk_factors, hidden_size=num_rnn_hidden_units, num_layers=num_rnn_layers, batch_first=True, **factory_kwargs)
            input_size = num_rnn_hidden_units
        elif rnn_type == enums.RNNType.GRU:
            self.rnn = nn.GRU(input_size=number_risk_factors, hidden_size=num_rnn_hidden_units, num_layers=num_rnn_layers, batch_first=True, **factory_kwargs)
            input_size = num_rnn_hidden_units
        else:
            raise ValueError("Unsupported rnn_type: {}".format(rnn_type))

        # Feed-forward head maps each timestep feature vector to a scalar CVA value.
        deep_layers = []
        for _ in range(num_deep_layers):
            deep_layers.append(nn.Linear(input_size, deep_hidden_units, **factory_kwargs))
            deep_layers.append(nn.ReLU())
            input_size = deep_hidden_units
        self.deep = nn.Sequential(*deep_layers)

        self.output_layer = nn.Linear(input_size, 1, **factory_kwargs)

    def forward(self, x):
        # x shape: (batch_size, time_steps, number_risk_factors)
        if self.rnn is None:
            sequence_features = x
        else:
            sequence_features, _ = self.rnn(x)

        # Keep every timestep: this is many-to-many, not many-to-one.
        deep_out = self.deep(sequence_features)
        output = self.output_layer(deep_out)
        # output shape: (batch_size, time_steps), matching the dataset target y.
        return output.squeeze(-1)
       
