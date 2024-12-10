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
        assert len(x.shape) == 3, "PatchMerging2D only accepts [c, h, w] "
        f"tensors. Received rank {len(x.shape)} tensor with shape {x.shape}"
        c, h, w = x.shape
        assert c == self.d_input, "Expected channel count to be "
        f"{self.d_input}. Received tensor with shape {x.shape}"

        h_out = h // 2
        w_out = w // 2
        h_crop = h_out * 2
        w_crop = w_out * 2

        # We crop the image to the largest even dimensions that don't require
        # padding.
        # Unlike MedMamba, we have no constraints on the heights and widths
        # coming into this layer, so we crop by default.
        x0 = x[:, 0:h_crop    :2, 0:w_crop    :2]
        x1 = x[:, 1:h_crop + 1:2, 0:w_crop    :2]
        x2 = x[:, 0:h_crop    :2, 1:w_crop + 1:2]
        x3 = x[:, 1:h_crop + 1:2, 1:w_crop + 1:2]
        assert x0.shape == (c, h_out, w_out)
        assert x1.shape == (c, h_out, w_out)
        assert x2.shape == (c, h_out, w_out)
        assert x3.shape == (c, h_out, w_out)

        x = torch.cat([x0, x1, x2, x3], 0)
        assert x.shape == (C * 4, h_out, w_out)

        x = x.permute(1, 2, 0)
        assert x.shape == (h_out, w_out, c * 4)

        x = self.norm(x)
        assert x.shape == (h_out, w_out, c * 4)

        x = self.reduction(x)
        assert x.shape == (h_out, w_out, c * 2)

        x = x.permute(2, 0, 1)
        assert x.shape == (c * 2, h_out, w_out)

        return x
