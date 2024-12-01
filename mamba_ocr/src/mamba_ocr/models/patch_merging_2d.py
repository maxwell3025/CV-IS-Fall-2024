import torch
from torch import nn

class PatchMerging2D(nn.Module):
    """Patch merging layer adapted from MedMamba.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.d_input = dim

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, "PatchMerging2D only accepts [C, H, W] "
        f"tensors. Received rank {len(x.shape)} tensor with shape {x.shape}"
        C, H, W = x.shape
        assert C == self.d_input, "Expected channel count to be "
        f"{self.d_input}. Received tensor with shape {x.shape}"

        H_out = H // 2
        W_out = W // 2
        H_crop = H_out * 2
        W_crop = W_out * 2

        # We crop the image to the largest even dimensions that don't require
        # padding.
        # Unlike MedMamba, we have no constraints on the heights and widths
        # coming into this layer, so we crop by default.
        x0 = x[:, 0:H_crop    :2, 0:W_crop    :2]
        x1 = x[:, 1:H_crop + 1:2, 0:W_crop    :2]
        x2 = x[:, 0:H_crop    :2, 1:W_crop + 1:2]
        x3 = x[:, 1:H_crop + 1:2, 1:W_crop + 1:2]
        assert x0.shape == (C, H_out, W_out)
        assert x1.shape == (C, H_out, W_out)
        assert x2.shape == (C, H_out, W_out)
        assert x3.shape == (C, H_out, W_out)

        x = torch.cat([x0, x1, x2, x3], 0)
        assert x.shape == (C * 4, H_out, W_out)

        x = x.permute(1, 2, 0)
        assert x.shape == (H_out, W_out, C * 4)

        x = self.norm(x)
        assert x.shape == (H_out, W_out, C * 4)

        x = self.reduction(x)
        assert x.shape == (H_out, W_out, C * 2)

        x = x.permute(2, 0, 1)
        assert x.shape == (C * 2, H_out, W_out)

        return x
