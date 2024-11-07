import mamba_ssm
import torch
from torch import nn

class SimpleMAMBA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        d_input: int,
        d_output: int,
    ):
        """This is a simple wrapper for a MAMBA layer.

        This model takes tensors of the shape (B, L, I), and inputs tensors of
        the shape (B, L, O), where
        - B represents the batch size.
        - L represents the sequence length.
        - I represents the input token size.
        - O represents the output token size.

        Note that this layer is only designed to take batched inputs, meaning
        rank-3 tensors.

        Args:
            d_model: An integer representing the number of channels that this
                model has.
            d_state: An integer representing the size of the internal state of
                each channel.
            d_conv: An integer representing the kernel size of the pre-ssm
                convolution.
            d_input: An integer representing the size of each input token.
            d_output: An integer representing the size of each output token.
        """
        super(SimpleMAMBA, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.d_state = d_state
        self.d_output = d_output
        self.d_conv = d_conv
        self.fc1 = nn.Linear(self.d_input, self.d_model)
        self.layer = mamba_ssm.Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )
        self.fc2 = nn.Linear(self.d_model, self.d_output)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape == (batch_size, length, self.d_input)

        x = self.fc1(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.layer(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.fc2(x)
        assert x.shape == (batch_size, length, self.d_output)
        
        return x