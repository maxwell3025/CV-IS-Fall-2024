import mamba_ssm
from mamba_ssm.modules import mamba_simple
import torch
from torch import nn
from . import drop


class SimpleMAMBA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        d_input: int,
        d_output: int,
        drop_path: float,
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
        self.drop_path = drop.DropPath(drop_prob=drop_path)
        self.fc2 = nn.Linear(self.d_model, self.d_output)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape == (batch_size, length, self.d_input)

        x = self.fc1(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.layer(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.drop_path(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.fc2(x)
        assert x.shape == (batch_size, length, self.d_output)
        
        return x

    def forward_debug(self, x: torch.Tensor):
        batch_size = x.shape[0]
        length = x.shape[1]
        assert x.shape == (batch_size, length, self.d_input)

        x = self.fc1(x)
        assert x.shape == (batch_size, length, self.d_model)

        x, dt = self.get_debug_info(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.drop_path(x)
        assert x.shape == (batch_size, length, self.d_model)

        x = self.fc2(x)
        assert x.shape == (batch_size, length, self.d_output)
        
        return x, dt
    
    def get_debug_info(self, hidden_states: torch.Tensor):
        """
        Evaluates the mamba layer and outputs debugging info.
        
        This is literally copied from the repo, so this should not be modified.
        
        hidden_states: (B, L, D)
        Returns: dt
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        # We do matmul and transpose BLH -> HBL at the same time
        xz = mamba_simple.rearrange(
            self.layer.in_proj.weight @ mamba_simple.rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.layer.in_proj.bias is not None:
            xz = xz + mamba_simple.rearrange(self.layer.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.layer.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.layer.d_conv :], it will error if seqlen < self.layer.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.layer.d_conv, and truncate otherwise.
            conv_state.copy_(mamba_simple.F.pad(x, (self.layer.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if mamba_simple.causal_conv1d_fn is None:
            x = self.layer.act(self.layer.conv1d(x)[..., :seqlen])
        else:
            assert self.layer.activation in ["silu", "swish"]
            x = mamba_simple.causal_conv1d_fn(
                x=x,
                weight=mamba_simple.rearrange(self.layer.conv1d.weight, "d 1 w -> d w"),
                bias=self.layer.conv1d.bias,
                activation=self.layer.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.layer.x_proj(mamba_simple.rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.layer.dt_rank, self.layer.d_state, self.layer.d_state], dim=-1)
        dt = self.layer.dt_proj.weight @ dt.t()
        dt = mamba_simple.rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = mamba_simple.rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = mamba_simple.rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.layer.activation in ["silu", "swish"]
        y = mamba_simple.selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.layer.D.float(),
            z=z,
            delta_bias=self.layer.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y # type: ignore
            ssm_state.copy_(last_state)
        y = mamba_simple.rearrange(y, "b d l -> b l d")
        out = self.layer.out_proj(y)

        # Prepare and return internals
        dt = mamba_simple.rearrange(dt, "b d l -> b l d", l=seqlen)
        return out, dt