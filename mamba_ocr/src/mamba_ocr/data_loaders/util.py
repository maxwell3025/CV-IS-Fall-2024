import numpy
from PIL import Image
import torch

DEFAULT_HEIGHT=64

def get_position_tensor(
    w: int,
    h: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    positional_encoding_vectors: numpy.ndarray,
) -> torch.Tensor:
    x_tensor = (numpy.arange(w) + 0.5) * (x_max - x_min) / w + x_min
    x_tensor = x_tensor[numpy.newaxis, :]
    x_tensor = numpy.broadcast_to(x_tensor, (h, w))
    assert x_tensor.shape == (h, w)

    y_tensor = (numpy.arange(h) + 0.5) * (y_max - y_min) / h + y_min
    y_tensor = y_tensor[:, numpy.newaxis]
    y_tensor = numpy.broadcast_to(y_tensor, (h, w))
    assert y_tensor.shape == (h, w)

    positional_encoding = numpy.stack((x_tensor, y_tensor), axis=-1)
    assert positional_encoding.shape == (h, w, 2)

    positional_encoding = positional_encoding @ positional_encoding_vectors.transpose()
    assert positional_encoding.shape == (h, w, positional_encoding_vectors.shape[0])

    positional_encoding = numpy.concatenate([
        numpy.sin(positional_encoding),
        numpy.cos(positional_encoding),
    ], 2)
    assert positional_encoding.shape == (h, w, positional_encoding_vectors.shape[0] * 2)

    positional_encoding = torch.from_numpy(positional_encoding)
    positional_encoding = positional_encoding.permute((2, 0, 1))
    return positional_encoding
