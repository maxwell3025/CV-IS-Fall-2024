from abc import ABC, abstractmethod
import torch
from typing import NewType

class OcrTask(ABC):
    @property
    @abstractmethod
    def d_alphabet(self) -> int:
        pass

    @property
    @abstractmethod
    def d_color(self) -> int:
        pass

    @property
    def d_feature(self) -> int:
        return self.d_alphabet + self.d_color + self.d_positional_encoding

    @property
    @abstractmethod
    def d_positional_encoding(self) -> int:
        pass

    @abstractmethod
    def get_alphabet_index(self, char: str) -> int:
        pass
        
    @abstractmethod
    def get_index_alphabet(self, index: int) -> str:
        pass

    @abstractmethod
    def get_batch(
        self,
        batch_size: int,
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Retrieve a batch of in-context OCR tasks.

        Args:
            batch_size: The number of contexts to return.
            split: A string equal to one of "train", "val", "test", which
                determines the source of the data.

        Returns:
            A list of length `batch_size` of contexts.
            Each context is a tuple, `(features, labels)`.
            `features` is a list of tensors representing decorated images with the
            shape `[d_feature, height, width]`.
            `labels` is a list of tensors representing the labels for the images
            encoded as one-hot tensors with the shape `[length, d_alphabet]`
        """
        pass

    @abstractmethod
    def get_batch_no_context(
        self,
        batch_size: int,
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Retrieve a batch of in-context OCR tasks.

            This function behaves the same way as `get_batch`, except that it
            splits the contexts so that each individual image is in a separate
            context.

        Args:
            batch_size: The number of contexts to return.
            split: A string equal to one of "train", "val", "test", which
                determines the source of the data.

        Returns:
            A list of length `batch_size` of contexts.
            Each context is a tuple, `(features, labels)`.
            `features` is a list of tensors representing decorated images with the
            shape `[d_feature, height, width]`.
            `labels` is a list of tensors representing the labels for the images
            encoded as one-hot tensors with the shape `[length, d_alphabet]`
        """
        pass
