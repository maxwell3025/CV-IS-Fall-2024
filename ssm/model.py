from torch import nn
from torch import Tensor
from torch import linalg
import torch
import math

class NaiveSsmLayer(nn.Module):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\

    The input shape of this layer is expected to be `[..., B, L]`,

    Args:
        num_channels (int): The number of output dimensions
        state_dimensions (int): The number of dimensions in the internal state
    """
    def __init__(self, num_channels, state_dimensions) -> None:
        super(NaiveSsmLayer, self).__init__()
        
        self.num_channels = num_channels
        self.state_dimensions = state_dimensions

        self.A = nn.Parameter(torch.Tensor(state_dimensions, state_dimensions))
        # a list of column vectors that each correspond to a channel-to-state
        # function
        self.B = nn.Parameter(torch.Tensor(state_dimensions, num_channels))
        # a list of column vectors that each correspond to a transposed 
        # state-to-channel function
        self.C = nn.Parameter(torch.Tensor(state_dimensions, num_channels))

        self.log_dt = nn.Parameter(torch.tensor((-2.0,)))

    def forward(self, signal: Tensor):
        dt = torch.exp(self.log_dt)
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

class NaiveDiagonalSsmLayer(nn.Module):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\
    Ih this layer, the A matrix is hardcoded to be a diagonal matrix
        
    The input shape of this layer is expected to be `[..., B, L]`,

    Args:
        num_channels (int): The number of output dimensions
        state_dimensions (int): The number of dimensions in the internal state
    """
    def __init__(self, num_channels, state_dimensions) -> None:
        super(NaiveDiagonalSsmLayer, self).__init__()
        
        self.num_channels = num_channels
        self.state_dimensions = state_dimensions

        diagonal_matrix = torch.zeros((state_dimensions, state_dimensions))
        for i in range(state_dimensions):
            for j in range(state_dimensions):
                if i == j:
                    diagonal_matrix[i, j] = i / state_dimensions
        print(diagonal_matrix)
        self.A = nn.Parameter(-diagonal_matrix, requires_grad=False)
        # a list of column vectors that each correspond to a channel-to-state
        # function
        self.B = nn.Parameter(torch.Tensor(state_dimensions, num_channels))
        # a list of column vectors that each correspond to a transposed 
        # state-to-channel function
        self.C = nn.Parameter(torch.Tensor(state_dimensions, num_channels))

        self.log_dt = nn.Parameter(torch.tensor((-2.0,)))

    def forward(self, signal: Tensor):
        dt = torch.exp(self.log_dt)
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

class HippoSsmLayer(nn.Module):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\
    Ih this layer, the A matrix is hardcoded to be a HiPPO matrix
        
    The input shape of this layer is expected to be `[..., B, L]`,

    Args:
        num_channels (int): The number of output dimensions
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

        self.log_dt = nn.Parameter(torch.tensor((-2.0,)))

    def forward(self, signal: Tensor):
        dt = torch.exp(self.log_dt)
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
