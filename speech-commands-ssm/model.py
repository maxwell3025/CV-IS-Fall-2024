from torch import nn
from torch import Tensor
from torch import linalg
import torch
import math

class NaiveSsmLayer(nn.Module):
    """Applies an SSM step on a variable-length sequence of length `N`\\
    The input shape of this layer is expected to be `[..., N, in_dimensions]`,
    where `in_dimensions` is the number of input channels.\\
    The output shape is `[..., N, out_dimensions]`, where `out_dimensions` is
    the number of output channels

    Args:
        in_dimensions (int): The number of input dimensions
        out_dimensions (int): The number of output dimensions
        state_dimensions (int): The number of dimensions in the internal state
    """
    def __init__(self, in_dimensions, state_dimensions, out_dimensions) -> None:
        super(NaiveSsmLayer, self).__init__()
        
        self.in_dimensions = in_dimensions
        self.state_dimensions = state_dimensions
        self.out_dimensions = out_dimensions

        self.A = nn.Parameter(torch.Tensor(state_dimensions, state_dimensions))
        self.B = nn.Parameter(torch.Tensor(state_dimensions, in_dimensions))
        self.C = nn.Parameter(torch.Tensor(out_dimensions, state_dimensions))

    def forward(self, signal: Tensor, dt: float):
        if signal.shape[-1] != self.in_dimensions:
            raise Exception(f"Received shape {signal.shape}, expected second to"
                            f" last index to be {self.in_dimensions}.")

        batch_dims = signal.shape[:-2]

        device = next(self.parameters()).device

        A_bar = torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions) - (dt * 0.5) * self.A
            ),
            torch.eye(self.state_dimensions) + (dt * 0.5) * self.A
        )
        B_bar = dt * torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions) - (dt * 0.5) * self.A
            ),
            self.B
        )
        C_bar = self.C

        state = torch.zeros(batch_dims + (self.state_dimensions, 1)).to(device)
        output_signal = []

        signal_length = signal.shape[-2]

        for i in range(signal_length):
            state = torch.matmul(A_bar, state)
            state = state + torch.matmul(B_bar, signal[..., i:i+1, :].swapaxes(-2, -1))
            output_signal.append(torch.matmul(C_bar, state))
        
        return torch.cat(output_signal, -1).swapaxes(-1, -2)

class HippoSsmLayer(nn.Module):
    """Applies an SSM step on a variable-length sequence of length `N`\\
    The input shape of this layer is expected to be `[..., N, in_dimensions]`,
    where `in_dimensions` is the number of input channels.\\
    The output shape is `[..., N, out_dimensions]`, where `out_dimensions` is
    the number of output channels

    Args:
        in_dimensions (int): The number of input dimensions
        out_dimensions (int): The number of output dimensions
        state_dimensions (int): The number of dimensions in the internal state
    """
    def __init__(self, num_channels, state_dimensions) -> None:
        super(HippoSsmLayer, self).__init__()
        
        self.num_channels = num_channels
        self.state_dimensions = state_dimensions

        hippo_matrix = torch.zeros((state_dimensions, state_dimensions))
        for i in range(state_dimensions):
            for j in range(state_dimensions):
                if i > j:
                    hippo_matrix[i, j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
                if i == j:
                    hippo_matrix[i, j] = i + 1
        self.A = nn.Parameter(-hippo_matrix, requires_grad=False)
        # a list of column vectors that each correspond to a channel-to-state
        # function
        self.B = nn.Parameter(torch.Tensor(state_dimensions, num_channels))
        # a list of column vectors that each correspond to a transposed 
        # state-to-channel function
        self.C = nn.Parameter(torch.Tensor(state_dimensions, num_channels))

    def forward(self, signal: Tensor, dt: float):
        #signal has shape [..., B, L]
        if signal.shape[-2] != self.num_channels:
            raise Exception(f"Received shape {signal.shape}, expected second to"
                            f" last index to be {self.num_channels}.")

        batch_dims = signal.shape[:-2]

        device = next(self.parameters()).device

        A_bar = torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions) - (dt * 0.5) * self.A
            ),
            torch.eye(self.state_dimensions) + (dt * 0.5) * self.A
        )
        B_bar = dt * torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions) - (dt * 0.5) * self.A
            ),
            self.B
        )
        C_bar = self.C

        state = torch.zeros(batch_dims + (self.state_dimensions, self.num_channels)).to(device)
        output_signal = []

        signal_length = signal.shape[-1]

        for i in range(signal_length):
            state = torch.matmul(A_bar, state)
            state = state + torch.mul(B_bar, signal[..., :, i:i+1].swapaxes(-1, -2))
            matchings = torch.mul(C_bar, state)
            output_signal.append(matchings.sum(-2, keepdim=True).swapaxes(-1, -2))
        
        full_output_signal = torch.cat(output_signal, -1)
        assert full_output_signal.shape[-1] == signal.shape[-1]
        assert full_output_signal.shape[-2] == signal.shape[-2]
        return full_output_signal
