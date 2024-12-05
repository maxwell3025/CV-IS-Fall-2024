from . import vanilla_vss
from functools import partial
import logging
import random
from timm.models.layers import DropPath
from torch import Tensor
import torch
from torch import nn
from typing import Callable
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    num_channels, height, width, = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(groups, channels_per_group, height, width)

    x = x.transpose(0, 1).contiguous()

    # flatten
    x = x.view(-1, height, width)

    return x

class SsConvSsm(nn.Module):
    """An implementation of the SS-Conv-SSM layer from MedMamba with context.

    A detailed diagram of the architecture can be found at
    [MedMamba: Vision Mamba for Medical Image Classification](https://arxiv.org/html/2403.03849v5#S3.F1).
    """
    def __init__(
        self,
        d_hidden: int,
        d_label: int,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        """Initialize an instance of SsConvSsm

        Args:
            d_hidden: Equal to both the input and output dimension for images.
            d_label: Equal to the number of channels in the labels
            drop_path: Probability that the Mamba layer is ignored. Defaults to
                0.
            norm_layer: A constructor for the normalization layer. Defaults to
                `partial(nn.LayerNorm, eps=1e-6)`.
            attn_drop_rate: The dropout probability for the Mamba layer.
                Defaults to 0.
            d_state: The internal state size for the Mamba Layer. Defaults to
                16.
        """
        super().__init__()
        self.ln_1 = norm_layer(d_hidden//2)
        self.self_attention = vanilla_vss.VanillaVss(
            d_feature=d_hidden//2,
            d_label=d_label,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs,
        )
        self.drop_path = drop_path

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(d_hidden // 2),
            nn.Conv2d(
                in_channels=d_hidden // 2,
                out_channels=d_hidden // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(d_hidden // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_hidden // 2,
                out_channels=d_hidden // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(d_hidden // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_hidden // 2,
                out_channels=d_hidden // 2,
                kernel_size=1,
                stride=1
            ),
            nn.ReLU()
        )

    def forward(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        input_left: list[torch.Tensor] = []
        input_right: list[torch.Tensor] = []
        for left, right in (feature.chunk(2,dim=0) for feature in features):
            input_left.append(left)
            input_right.append(right)

        input_right = [input_right_.permute(1, 2, 0) for input_right_ in input_right]
        input_right = [self.ln_1(input_right_) for input_right_ in input_right]
        input_right = [input_right_.permute(2, 0, 1) for input_right_ in input_right]
        input_right = self.self_attention(input_right, labels)
        if (random.random() < self.drop_path) and self.training:
            input_right = [input_right_ * 0 for input_right_ in input_right]

        input_left = [self.conv33conv33conv11(input_left_.unsqueeze(0)).squeeze(0) for input_left_ in input_left]

        output = [torch.cat((input_left_,input_right_),dim=0) for input_left_, input_right_ in zip(input_left, input_right)]
        output = [channel_shuffle(output_,groups=2) for output_ in output]
        return [feature + output_ for feature, output_ in zip(features, output)]
