from . import sequence_stack
import torch
from torch import nn

class SequenceStackCnnConfig(sequence_stack.SequenceStackConfig):
    def __init__(self, config: dict[str, any]) -> None:
        super(SequenceStackCnnConfig, self).__init__(config)
        self.conv_n_layer = int(config["conv_n_layer"])
        self.conv_d_input = int(config["conv_d_input"])
        self.conv_d_intermediate = int(config["conv_d_intermediate"])
        self.conv_d_output = int(config["conv_d_input"])
        
class SequenceStackCnn(sequence_stack.SequenceStack):
    def __init__(
        self,
        config: SequenceStackCnnConfig,
    ) -> None:
        super(SequenceStackCnn, self).__init__(config)
        self.cnn_layers = nn.ModuleList
    
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
    
