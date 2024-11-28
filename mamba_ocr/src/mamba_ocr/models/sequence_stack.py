from . import simple_lstm
from . import simple_mamba
from . import ocr_model
import logging
import mamba_ssm
import torch
from torch import nn
from typing import Any
from . import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class SequenceStackConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.n_layer = int(kwargs["n_layer"])
        self.d_input = int(kwargs["d_input"])
        self.d_output = int(kwargs["d_output"])
        self.d_intermediate = int(kwargs["d_intermediate"])
        self.skip_connection = bool(kwargs["skip_connection"])

        self.lstm_layer_idx = list(kwargs["lstm_layer_idx"])
        self.lstm_d_hidden = int(kwargs["lstm_d_hidden"])

        self.mamba_d_model = int(kwargs["mamba_d_model"])
        self.mamba_d_state = int(kwargs["mamba_d_state"])
        self.mamba_d_conv = int(kwargs["mamba_d_conv"])

class SequenceStack(ocr_model.OcrModel):
    """TODO format this docstring

    This model is meant to just run images in normal sequence order through the
    model
    """
    def __init__(
        self,
        **kwargs
    ) -> None:
        super(SequenceStack, self).__init__()

        self.config = SequenceStackConfig(**kwargs)
        layers = []
        for i in range(self.config.n_layer):
            if i in self.config.lstm_layer_idx:
                layers.append(
                    simple_lstm.SimpleLSTM(
                        input_dim=self.config.d_intermediate,
                        hidden_dim=self.config.lstm_d_hidden,
                        output_dim=self.config.d_intermediate,
                    )
                )
            else:
                layers.append(
                    simple_mamba.SimpleMAMBA(
                        d_input=self.config.d_intermediate,
                        d_output=self.config.d_intermediate,
                        d_model=self.config.mamba_d_model,
                        d_conv=self.config.mamba_d_conv,
                        d_state=self.config.mamba_d_state,
                    )
                )
        self.fc1 = nn.Linear(self.config.d_input, self.config.d_intermediate)
        self.layers = nn.ModuleList(layers)
        self.fc2 = nn.Linear(self.config.d_intermediate, self.config.d_output)
    
    def forward(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        # The input must be a rank 3 tensor
        def sequencify(image: torch.Tensor) -> torch.Tensor:
            assert len(image.shape) == 3
            return image.transpose(1, 2).flatten(1, 2).transpose(0, 1)
        
        features = list(map(sequencify, features))
        x = util.inject_sequence_labels(features, labels)
        x = torch.unsqueeze(x, 0)
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

        index = -1
        output = []
        for i in range(len(features)):
            len_feature = features[i].shape[0]
            len_label = labels[i].shape[0]
            index += len_feature
            output.append(x[0, index:index + labels[i].shape[0], :])
            index += len_label
        
        return torch.cat(output)
    
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