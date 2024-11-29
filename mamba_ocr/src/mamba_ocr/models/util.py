import torch

def image_to_sequence_ss2d(image: torch.Tensor):
    """Converts a [C, H, W] image into 4 sequences according to SS2D.

    Args:
        image: A PyTorch tensor with the shape [C, H, W]

    Returns:
        A tuple (a, b, c, d), each of which represents is a tensor with the
        shape [H * W, C].
        a is row major order.
        b is reverse row major order.
        c is column major order.
        d is reverse row major order.
    """
    assert len(image.shape) == 3
    C, H, W = image.shape

    a = image.flatten(1, 2)
    assert a.shape == (C, H*W)

    b = a.flip(1)
    assert b.shape == (C, H*W)

    c = image.transpose(1, 2).flatten(1, 2)
    assert c.shape == (C, H*W)

    d = c.flip(1)
    assert d.shape == (C, H*W)

    return a, b, c, d

def sequence_to_image_ss2d(
    sequences: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    H: int,
    W: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Converts [L, D] Sequences back into the [C, H, W] sequences from the inverse.

    Args:
        sequences: _description_

    Returns:
        _description_
    """
    a, b, c, d = sequences
    return (
        a.unflatten(0, (H, W)).permute(2, 0, 1),
        b.flip(dims=(0,)).unflatten(0, (H, W)).permute(2, 0, 1),
        c.unflatten(0, (W, H)).transpose(1, 2).permute(2, 0, 1),
        d.flip(dims=(0,)).unflatten(0, (W, H)).transpose(1, 2).permute(2, 0, 1),
    )

def inject_sequence_labels(
    features: list[torch.Tensor],
    labels: list[torch.Tensor],
):
    """Append one-hot label sequences to the end of a sequence.
    
    Example:
    ```python
        import torch
        sequence = [
            torch.tensor([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
            torch.tensor([
                [3, 2, 1],
                [6, 5, 4],
                [9, 8, 7],
            ]),
        ]
        labels = [
            torch.tensor([
                [0, 1],
                [0, 0],
                [0, 1],
            ]),
            torch.tensor([
                [1, 0],
                [0, 1],
                [0, 0],
            ]),
        ]
        inject_sequence_labels(sequence, labels)
        # torch.Tensor([
        #     [1, 2, 3, 0, 0],
        #     [4, 5, 6, 0, 0],
        #     [7, 8, 9, 0, 0],
        #     [0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 1],
        #     [3, 2, 1, 0, 0],
        #     [6, 5, 4, 0, 0],
        #     [9, 8, 7, 0, 0],
        #     [0, 0, 0, 1, 0],
        #     [0, 0, 0, 0, 1],
        #     [0, 0, 0, 0, 0],
        # ])
    ```

    Args:
        sequence (list[torch.Tensor]): _description_
        labels (list[torch.Tensor]): _description_
    """
    assert len(features) == len(labels)
    assert len(features[0].shape) == 2
    assert len(labels[0].shape) == 2
    d_feature = features[0].shape[1]
    d_label = labels[0].shape[1]
    # This is a list of tensors where each tensor is a contiguous section of
    # data or label padded with zeroes
    sections = []
    for i in range(len(features)):
        feature = features[i]
        label = labels[i]
        assert feature.shape[1] == d_feature
        assert label.shape[1] == d_label

        feature_padding = torch.zeros((feature.shape[0], d_label), device=feature.device)
        label_padding = torch.zeros((label.shape[0], d_feature), device=label.device)

        feature = torch.cat((feature, feature_padding), dim=1)
        label = torch.cat((label_padding, label), dim=1)
        assert feature.shape[1] == d_feature + d_label
        assert label.shape[1] == d_feature + d_label

        sections += [feature, label]

    return torch.cat(sections, dim=0)
