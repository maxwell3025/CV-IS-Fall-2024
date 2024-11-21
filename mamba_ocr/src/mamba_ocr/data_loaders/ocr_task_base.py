from . import ocr_task
import numpy
import torch

class OcrTaskBase(ocr_task.OcrTask):
    def __init__(
        self,
        features: list[list[torch.Tensor]],
        masks: list[list[torch.Tensor]],
        split: tuple[float, float]
    ):
        train_size = round(len(features) * split[0])
        test_size = round(len(features) * split[1]) - train_size
        val_size = len(features) - test_size
        self.features = dict(
            train=features[:train_size],
            test=features[train_size:train_size + test_size],
            val=features[-val_size:],
        )
        self.masks = dict(
            train=masks[:train_size],
            test=masks[train_size:train_size + test_size],
            val=masks[-val_size:],
        )
        self.current_indices = dict(
            train=(0, 0),
            test=(0, 0),
            val=(0, 0),
        )
    
    def get_batch(
        self,
        batch_size: int,
        pad_length: int,
        split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features[split]
        masks = self.masks[split]
        current_context, current_word = self.current_indices[split]
        if(batch_size > len(features) - current_context):
            current_context = 0
            current_word = 0
        if(current_word != 0):
            current_context += 1
            current_word = 0

        features = features[current_context:current_context+batch_size]
        masks = masks[current_context:current_context+batch_size]
        current_context += batch_size
        self.current_indices[split] = current_context, current_word
        
        padded_features = []
        padded_labels = []
        padded_masks = []
        for i in range(batch_size):
            full_features = torch.cat(features[i])
            full_mask = torch.cat(masks[i])
            overall_len = full_features.shape[0]
            if overall_len < pad_length + 1:
                extra_len = pad_length - overall_len + 1
                full_features = torch.cat(full_features, torch.zeros((extra_len, self.get_d_input())))
                full_mask = torch.cat(full_mask, torch.zeros((extra_len,)))
            full_labels = full_features[:, -self.get_d_alphabet():]

            full_features = full_features[:pad_length, :]
            full_labels = full_labels[1:pad_length + 1, :]
            full_mask = full_mask[1:pad_length + 1]

            padded_features.append(full_features)
            padded_labels.append(full_labels)
            padded_masks.append(full_mask)
        return torch.stack(padded_features), torch.stack(padded_labels), torch.stack(padded_labels),
        
    def get_batch_no_context(
        self,
        batch_size: int,
        pad_length: int,
        split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_features = self.features[split]
        full_masks = self.masks[split]
        current_context, current_word = self.current_indices[split]
        features: list[list[torch.Tensor]] = []
        masks: list[list[torch.Tensor]] = []
        for i in range(batch_size):
            features.append([full_features[current_context][current_word]])
            masks.append([full_masks[current_context][current_word]])
            current_word += 1
            if current_word >= len(full_features[current_context]):
                current_word = 0
                current_context += 1
            if current_context >= len(full_features):
                current_context = 0
        self.current_indices[split] = current_context, current_word
        
        padded_features = []
        padded_labels = []
        padded_masks = []
        for i in range(batch_size):
            full_features = torch.cat(features[i])
            full_mask = torch.cat(masks[i])
            overall_len = full_features.shape[0]
            if overall_len < pad_length + 1:
                extra_len = pad_length - overall_len + 1
                full_features = torch.cat(full_features, torch.zeros((extra_len, self.get_d_input())))
                full_mask = torch.cat(full_mask, torch.zeros((extra_len,)))
            full_labels = full_features[:, -self.get_d_alphabet():]

            full_features = full_features[:pad_length, :]
            full_labels = full_labels[1:pad_length + 1, :]
            full_mask = full_mask[1:pad_length + 1]

            padded_features.append(full_features)
            padded_labels.append(full_labels)
            padded_masks.append(full_mask)
        return torch.stack(padded_features), torch.stack(padded_labels), torch.stack(padded_labels),

    def get_position_tensor(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> torch.Tensor:
        x_tensor = numpy.arange(w) + x
        x_tensor = x_tensor[numpy.newaxis, :]
        x_tensor = numpy.broadcast_to(x_tensor, (h, w))
        assert x_tensor.shape == (h, w)

        y_tensor = numpy.arange(h) + y
        y_tensor = y_tensor[:, numpy.newaxis]
        y_tensor = numpy.broadcast_to(y_tensor, (h, w))
        assert y_tensor.shape == (h, w)

        positional_encoding = numpy.stack((x_tensor, y_tensor), axis=-1)
        assert positional_encoding.shape == (h, w, 2)

        positional_encoding = positional_encoding @ self.positional_encoding_vectors.transpose()
        assert positional_encoding.shape == (h, w, self.get_d_positional_encoding() / 2)

        positional_encoding = numpy.concatenate([
            numpy.sin(positional_encoding),
            numpy.cos(positional_encoding),
        ], 2)
        assert positional_encoding.shape == (h, w, self.get_d_positional_encoding())

        positional_encoding: torch.Tensor = torch.from_numpy(positional_encoding)
        positional_encoding = positional_encoding.permute((2, 0, 1))
        return positional_encoding
    