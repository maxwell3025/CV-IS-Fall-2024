from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch import nn
from cfg_ops_mamba.config import MambaLstmConfig

class CompatibleLstm(nn.LSTM):
    """The default pytorch LSTM modified to fit the block interface for MAMBA.
    """
    def __init__(self, **kwargs):
        super(CompatibleLstm, self).__init__(
            batch_first=True,
            **kwargs,
        )
        self.linear = nn.Linear(kwargs["hidden_size"], kwargs["input_size"])
    def forward(self, x, residual, **kwargs):
        output, _ = super(CompatibleLstm, self).forward(x)
        output = self.linear(output)
        return output, residual
        
def MambaLMHeadModelLstm(config: MambaLstmConfig, device: any):
    """Create a MambaLMHeadModel with LSTM layers injected into it.

    Args:
        config: An object containing all of the hyperparameters for this model.
        device: The device that this model is on

    Returns:
        A MambaLMHeadModel with injected LSTM Layers
    """
    model = MambaLMHeadModel(config=config, device=device)
    for index in config.lstm_layer_idx:
        if index < 0 or index >= len(model.backbone.layers):
            print(f"Warning: attempted to set layer {index} to an"
                   "LSTM, which is out of bounds.")
            continue
        my_lstm = CompatibleLstm(
            input_size=config.d_model,
            **config.lstm_cfg,
        ).to(device)
        model.backbone.layers[index] = my_lstm
    return model