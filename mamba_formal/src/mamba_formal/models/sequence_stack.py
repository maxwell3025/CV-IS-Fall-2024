from mamba_formal.models import simple_lstm
from mamba_formal.models import simple_mamba
import torch
from torch import nn
from typing import Any

class SequenceStackConfig:
    def __init__(self, config: dict[str, Any]) -> None:
        self.n_layer = int(config["n_layer"])
        self.d_input = int(config["d_input"])
        self.d_output = int(config["d_output"])
        self.d_intermediate = int(config["d_intermediate"])
        self.skip_connection = bool(config["skip_connection"])

        self.lstm_layer_idx = list(config["lstm_layer_idx"])
        self.lstm_d_hidden = int(config["lstm_d_hidden"])

        self.mamba_d_model = int(config["mamba_d_model"])
        self.mamba_d_state = int(config["mamba_d_state"])
        self.mamba_d_conv = int(config["mamba_d_conv"])
        self.mamba_drop_prob = 0 if "mamba_drop_prob" not in config else config["mamba_drop_prob"]

class SequenceStack(nn.Module):
    def __init__(
        self,
        config: SequenceStackConfig,
    ) -> None:
        super(SequenceStack, self).__init__()

        self.config = config
        layers = []
        for i in range(config.n_layer):
            if i in config.lstm_layer_idx:
                layers.append(
                    simple_lstm.SimpleLSTM(
                        input_dim=config.d_intermediate,
                        hidden_dim=config.lstm_d_hidden,
                        output_dim=config.d_intermediate,
                    )
                )
            else:
                layers.append(
                    simple_mamba.SimpleMAMBA(
                        d_input=config.d_intermediate,
                        d_output=config.d_intermediate,
                        d_model=config.mamba_d_model,
                        d_conv=config.mamba_d_conv,
                        d_state=config.mamba_d_state,
                        drop_path=config.mamba_drop_prob,
                    )
                )
        self.fc1 = nn.Linear(config.d_input, config.d_intermediate)
        self.layers = nn.ModuleList(layers)
        self.fc2 = nn.Linear(config.d_intermediate, config.d_output)
    
    def forward(self, x: torch.Tensor, num_last_tokens=None):
        # The input must be a rank 3 tensor
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape[2] == self.config.d_input

        x = self.fc1(x)
        assert x.shape == (batch_size, length, self.config.d_intermediate)

        for layer in self.layers:
            x_new = layer(x)
            assert x_new.shape == (batch_size, length,
                                   self.config.d_intermediate)

            if self.config.skip_connection:
                x = x + x_new
            else:
                x = x_new

        x = self.fc2(x)
        assert x.shape == (batch_size, length, self.config.d_output)

        if num_last_tokens != None:
            x = x[:, -num_last_tokens:, :]
        return x
    
    def forward_debug(self, x: torch.Tensor, num_last_tokens=None):
        dt_info: list[torch.Tensor] = []

        # The input must be a rank 3 tensor
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape[2] == self.config.d_input

        x = self.fc1(x)
        assert x.shape == (batch_size, length, self.config.d_intermediate)

        for layer in self.layers:
            if isinstance(layer, simple_mamba.SimpleMAMBA):
                x_new, dt = layer.forward_debug(x)
                dt_info.append(dt)
            else:
                x_new = layer(x)
            assert x_new.shape == (batch_size, length,
                                   self.config.d_intermediate)

            if self.config.skip_connection:
                x = x + x_new
            else:
                x = x_new

        x = self.fc2(x)
        assert x.shape == (batch_size, length, self.config.d_output)

        if num_last_tokens != None:
            x = x[:, -num_last_tokens:, :]
        return x, dt_info
    
