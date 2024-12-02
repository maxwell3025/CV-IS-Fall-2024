from . import util
from einops import rearrange, repeat
import logging
import math
import torch
from torch import nn
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class VanillaVss(nn.Module):
    """Modified from MedMamba
    
    This encompasses the entire SSM-Branch in the
    [MedMamba architecture](https://arxiv.org/html/2403.03849v5#S3.F1).

    This is the vanilla VSS block described in
    [VMamba: Visual State Space Model](https://arxiv.org/html/2401.10166v3#S4.F3)
    (See part (c) of the figure).
    """
    def __init__(
        self,
        d_feature: int,
        d_label: int,
        d_state=16,
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
        """Initializes an instance of SsConvSsm

        Args:
            d_model: The number of image input channels.
            d_label: the number of channels in the labels.
            d_state: The size of each instance of the internal state. Defaults
                to 16.
            d_conv: The kernel size for the convolution kernel. This must be an
                odd number. Defaults to 3.
            expand: The amount the expand the input channel count before feeding
                to the internal SSM layer. Defaults to 2.
            dt_rank: The bottleneck size for dt. Defaults to "auto".
            dt_min: The minimum initial value for dt's bias. Defaults to 0.001.
            dt_max: The maximum initial value for dt's bias. Defaults to 0.1.
            dt_init: The method for initializing the projection for dt. Can be
                one of: "constant", "random". Defaults to "random".
            dt_scale: The scale of the dt projection. Defaults to 1.0.
            dt_init_floor: An additional floor on dt's bias. Defaults to 1e-4.
            dropout: A dropout likelihood for a dropout appleid at the end.
                Defaults to 0.
            conv_bias: Equal to `True` if the convolution layer should have
                bias. Defaults to True.
            bias: Equal to `True` if the output projection should have bias.
                Defaults to False.
            device: The pytorch device that all layers should be on. Defaults to
                None.
            dtype: The dtype that all calculations should use. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_feature = d_feature
        self.d_label = d_label
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_feature)
        self.dt_rank: int = math.ceil(self.d_feature / 16) if dt_rank == "auto" else dt_rank # type: ignore

        self.in_proj = nn.Linear(self.d_feature, self.d_inner * 2, bias=bias, **factory_kwargs)
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
            nn.Linear(self.d_inner + self.d_label, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner + self.d_label, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner + self.d_label, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner + self.d_label, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner + self.d_label, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner + self.d_label, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner + self.d_label, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner + self.d_label, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner + self.d_label, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner + self.d_label, copies=4, merge=True) # (K=4, D, N)

        self.pre_scale_proj = nn.Linear(self.d_inner + self.d_label, self.d_inner, bias=False, **factory_kwargs)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_feature, bias=bias, **factory_kwargs)
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

        B = 1
        xs = torch.stack([s1, s2, s3, s4])
        xs = xs.unsqueeze(0)
        xs = xs.transpose(2, 3)
        D = xs.shape[2]
        L = xs.shape[3]


        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.contiguous().float().view(B, -1, L) # (b, k * d, l)
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
        a = a.transpose(0, 1)
        b = b.transpose(0, 1)
        c = c.transpose(0, 1)
        d = d.transpose(0, 1)
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
        assert len(features[0].shape) == 3, ("Expected features to be a list "
        "of [C, H, W] tensors. Received a tensor with shape "
        f"{features[0].shape}.")

        C, H_0, W_0 = features[0].shape
        assert C == self.d_feature, (f"Expected images with {self.d_feature} "
        f"channels. Received a tensor with shape {features[0].shape}")

        assert len(labels[0].shape) == 2, ("Expected labels to be a list of "
        f"[L, D] tensors. Received a tensor with shape {labels[0].shape}")

        L_0, D = labels[0].shape
        assert D == self.d_label, (f"Expected labels to have {self.d_label} "
        f"channels. Received a tensor with shape {labels[0].shape}")

        features = [feature.permute(1, 2, 0) for feature in features]
        xz = [self.in_proj(feature) for feature in features]
        xz = [xz_.permute(2, 0, 1) for xz_ in xz]

        x, z = [], []
        for x_, z_ in (xz_.chunk(2, dim=0) for xz_ in xz):
            x.append(x_)
            z.append(z_)

        x = [self.act(self.conv2d(x_)) for x_ in x] # (c, h, w)
        y1, y2, y3, y4 = self.forward_core(x, labels)
        assert y1[0].dtype == torch.float32
        y = [y1_ + y2_ + y3_ + y4_ for y1_, y2_, y3_, y4_ in zip(y1, y2, y3, y4)]

        y = [y_.permute(1, 2, 0) for y_ in y]
        y = [self.pre_scale_proj(y_) for y_ in y]
        y = [y_.permute(2, 0, 1) for y_ in y]

        y = [y_.permute(1, 2, 0) for y_ in y]
        y = [self.out_norm(y_) for y_ in y]
        y = [y_.permute(2, 0, 1) for y_ in y]

        y = [y_ * nn.functional.silu(z_) for y_, z_ in zip(y, z)]

        y = [y_.permute(1, 2, 0) for y_ in y]
        out = [self.out_proj(y_) for y_ in y]
        out = [out_.permute(2, 0, 1) for out_ in out]

        if self.dropout is not None:
            out = [self.dropout(out_) for out_ in out]
        return out
