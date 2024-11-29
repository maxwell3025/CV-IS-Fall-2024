from einops import rearrange, repeat
from functools import partial
import math
from timm.models.layers import DropPath
from torch import Tensor
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, Callable
from . import util
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class SS2D(nn.Module):
    """Modified from MedMamba
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank: int = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # type: ignore

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs, # type: ignore
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True # type: ignore
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True # type: ignore
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True # type: ignore
        return D

    def forward_core(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        self.selective_scan = selective_scan_fn # type: ignore
        s1, s2, s3, s4 = [], [], [], []
        dims = []
        label_sizes = []
        for image in features:
            a, b, c, d = util.image_to_sequence_ss2d(image) 
            s1.append(a)
            s2.append(b)
            s3.append(c)
            s4.append(d)
            dims.append((image.shape[1], image.shape[2]))
        for label in labels:
            label_sizes.append(label.shape[0])
        s1 = util.inject_sequence_labels(s1, labels)
        s2 = util.inject_sequence_labels(s2, labels)
        s3 = util.inject_sequence_labels(s3, labels)
        s4 = util.inject_sequence_labels(s4, labels)
        K = 4

        xs = torch.stack([s1, s2, s3, s4])
        xs = xs.flip(2, 3)
        xs = xs.unsqueeze(0)
        B = 1
        L = xs.shape[3]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L) # type: ignore
        assert out_y.dtype == torch.float

        a, b, c, d = out_y[0, 0, :, :], out_y[0, 1, :, :], out_y[0, 2, :, :], out_y[0, 3, :, :]
        a = a.transpose()
        b = b.transpose()
        c = c.transpose()
        d = d.transpose()
        a_new, b_new, c_new, d_new = [], [], [], []
        index = 0
        for i in range(len(features)):
            H, W = dims[i]
            HW = H * W
            aa, bb, cc, dd = util.sequence_to_image_ss2d(
                (
                    a[index:index + HW],
                    b[index:index + HW],
                    c[index:index + HW],
                    d[index:index + HW],
                ),
                H,
                W,
            )
            a_new.append(aa)
            b_new.append(bb)
            c_new.append(cc)
            d_new.append(dd)
        return a_new, b_new, c_new, d_new

    def forward(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
        **kwargs,
    ):
        xz = [self.in_proj(x) for x in features]
        x, z = (list(seq) for seq in zip(xz_.chunk(2, dim=0) for xz_ in xz)) # (c, h, w)

        x = [x_.permute(0, 3, 1, 2).contiguous() for x_ in x]
        x = [self.act(self.conv2d(x_)) for x_ in x] # (c, h, w)
        y1, y2, y3, y4 = self.forward_core(x, labels)
        assert y1[0].dtype == torch.float32
        y = [y1_ + y2_ + y3_ + y4_ for y1_, y2_, y3_, y4_ in zip(y1, y2, y3, y4)]
        y = [self.out_norm(y_) for y_ in y]
        y = [y_ * nn.functional.silu(z_) for y_, z_ in zip(y, z)]
        out = [self.out_proj(y_) for y_ in y]
        if self.dropout is not None:
            out = [self.dropout(out_) for out_ in out]
        return out

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x

class SsConvSsm(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim//2)
        self.self_attention = SS2D(d_model=hidden_dim//2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim//2,out_channels=hidden_dim//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        input_left: list[torch.Tensor]
        input_right: list[torch.Tensor]
        input_left, input_right = [list(chunk) for chunk in zip(feature.chunk(2,dim=0) for feature in features)] # type: ignore

        input_right = [self.ln_1(input_right_) for input_right_ in input_right]
        input_right = self.self_attention(input_right, labels)
        input_right = [self.drop_path(input_right_) for input_right_ in input_right]

        input_left = [input_left_.permute(0,3,1,2).contiguous() for input_left_ in input_left]
        input_left = [self.conv33conv33conv11(input_left_) for input_left_ in input_left]
        input_left = [input_left_.permute(0,2,3,1).contiguous() for input_left_ in input_left]

        output = [torch.cat((input_left_,x_),dim=-1) for input_left_, x_ in zip(input_left, input_right)]
        output = [channel_shuffle(output_,groups=2) for output_ in output]
        return [feature + output_ for feature, output_ in zip(features, output)]
