from torch import nn
from torch import Tensor
from torch import linalg
import torch
import math
import numpy
from matplotlib import pyplot

class NaiveSsmBase(nn.Module):
    """Base class for all SSM models
    
    This implements the naive algorithm
    """

    def forward(self, sequence: Tensor):
        """Passes a sequence through this ssm

        Args:
            sequence (Tensor): The original sequence; has the shape
            `[..., D, L]`

        Returns:
            Tensor: The output sequence for the model
        """
        dt = torch.exp(self.log_dt)
        if sequence.shape[-2] != self.num_channels:
            raise Exception(f"Received shape {sequence.shape}, expected second to"
                            f" last index to be {self.num_channels}.")

        batch_dims = sequence.shape[:-2]

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

        signal_length = sequence.shape[-1]

        for i in range(signal_length):
            state = torch.matmul(A_bar, state)
            state = state + torch.mul(B_bar, sequence[..., :, i:i+1].swapaxes(-1, -2))
            matchings = torch.mul(C_bar, state)
            output_signal.append(matchings.sum(-2, keepdim=True).swapaxes(-1, -2))
        
        full_output_signal = torch.cat(output_signal, -1)
        assert full_output_signal.shape[-1] == sequence.shape[-1]
        assert full_output_signal.shape[-2] == sequence.shape[-2]
        return full_output_signal
    
class NaiveSsmLayer(NaiveSsmBase):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\

    The input shape of this layer is expected to be `[..., D, L]`,

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

class DiagonalSsmLayer(NaiveSsmBase):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\
    Ih this layer, the A matrix is hardcoded to be a diagonal matrix
        
    The input shape of this layer is expected to be `[..., D, L]`,

    Args:
        num_channels (int): The number of output dimensions
        state_dimensions (int): The number of dimensions in the internal state
    """
    def __init__(self, num_channels, state_dimensions) -> None:
        super(DiagonalSsmLayer, self).__init__()
        
        self.num_channels = num_channels
        self.state_dimensions = state_dimensions

        diagonal_matrix = torch.zeros((state_dimensions, state_dimensions))
        for i in range(state_dimensions):
            for j in range(state_dimensions):
                if i == j:
                    diagonal_matrix[i, j] = i / state_dimensions
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
        #signal has shape [..., D, L]
        if signal.shape[-2] != self.num_channels:
            raise Exception(f"Received shape {signal.shape}, expected second to"
                            f" last index to be {self.num_channels}.")

        batch_dims = signal.shape[:-2]

        device = next(self.parameters()).device

        A_bar = torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions, device=signal.device) - (dt * 0.5) * self.A
            ),
            torch.eye(self.state_dimensions, device=signal.device) + (dt * 0.5) * self.A
        )
        B_bar = dt * torch.matmul(
            linalg.inv(
                torch.eye(self.state_dimensions, device=signal.device) - (dt * 0.5) * self.A
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

class HippoSsmLayer(NaiveSsmBase):
    """This layer applies an arbitrary SSM convolution using an unoptimized
    algorithm.\\
    Ih this layer, the A matrix is hardcoded to be a HiPPO matrix
        
    The input shape of this layer is expected to be `[..., D, L]`,

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

class HippoSsmLayerTransposed(nn.Module):
    def __init__(self, D, N) -> None:
        super(HippoSsmLayerTransposed, self).__init__()
        self.layer = HippoSsmLayer(D, N)
    def forward(self, sequence: Tensor):
        return self.layer(sequence.transpose(-1, -2)).transpose(-1, -2)

class S6Layer(nn.Module):
    """This is the SSM layer for the Mamba architecture, outlined in section 3.2
    of
    [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)

    Since Mamba has a input-dependent step size, a custom forward function is
    needed.
    
    Also note that for now, the forward function uses the naive algorithm.

    The variable naming scheme follows the
    [paper](https://arxiv.org/pdf/2312.00752)'s naming scheme:
    - B: batch size
    - L: sequence length
    - D: channel count
    - N: single-instance state size
    """
    def __init__(self, D, N) -> None:
        super(S6Layer, self).__init__()
        self.D = D
        self.N = N
        A_init = -numpy.arange(1, D+1)
        A_init = numpy.broadcast_to(A_init[:, numpy.newaxis], (D, N))
        A_init = numpy.copy(A_init)
        self.A = nn.Parameter(torch.from_numpy(A_init).float()) # S4D-Real
        self.s_B = nn.Linear(D, N)
        self.s_C = nn.Linear(D, N)
        # Note that this gives a single timestep for all channels
        self.s_Delta = nn.Linear(D, 1)
        self.bias_Delta = nn.Parameter(torch.log(torch.rand((1,)) / (0.1 - 0.001) + 0.001))
        self.t_Delta = nn.Softplus()
    
    def forward(self, sequence: Tensor) -> Tensor:
        """This is a different implementation from the normal SSM layer

        Args:
            sequence (Tensor): This is the sequence of signals. The expected
            shape is [..., L, D]

        Returns:
            Tensor: The output sequence
        """
        # Initialize dimensions
        (L, D) = sequence.shape[-2:]
        batch_shape = sequence.shape[:-2]
        N = self.N
        assert(D == self.D)

        B: Tensor = self.s_B(sequence)
        assert(B.shape == (*batch_shape, L, N))

        C: Tensor = self.s_C(sequence)
        assert(C.shape == (*batch_shape, L, N))

        Delta: Tensor = self.t_Delta(self.s_Delta(sequence) + self.bias_Delta)
        Delta = Delta.broadcast_to((*batch_shape, L, D))
        # TODO broadcast Delta to the correct shape as detailed in the paper
        assert(Delta.shape == (*batch_shape, L, D))

        # print("Shape of Delta: ", Delta.shape)
        A_broadcasted = self.A.broadcast_to((*batch_shape, L, D, N))
        B_broadcasted = B.unsqueeze(-2).broadcast_to((*batch_shape, L, D, N))
        Delta_broadcasted = Delta.unsqueeze(Delta.dim()).broadcast_to((*batch_shape, L, D, N))
        A_bar: Tensor = torch.exp(A_broadcasted * Delta_broadcasted)
        B_bar: Tensor = self.A.reciprocal() * (A_bar - 1) * B_broadcasted

        current_state = torch.zeros((*batch_shape, D, N))
        outputs = []
        # Recurrent scan
        for time_step in range(L):
            current_input = sequence[..., time_step, :]
            assert(current_input.shape == (*batch_shape, D))
            current_state: Tensor = current_state * A_bar[..., time_step, :, :] + \
                current_input.unsqueeze(-1).broadcast_to((*batch_shape, D, N)) \
                * B_bar[..., time_step, :, :]
            outputs.append(
                (current_state * (C[..., time_step:time_step+1, :])) \
                .sum(-1) \
                .unsqueeze(-2)
            )
        self.previous_Delta = Delta
        self.previous_sequence = sequence
        self.previous_prediction = torch.cat(outputs, -2)
        return torch.cat(outputs, -2)
            

class MambaLayer(nn.Module):
    """This is the full block detailed in the Mamba paper.
    """
    def __init__(self) -> None:
        super(S6Layer, self).__init__()
    