from abc import abstractmethod
import torch
from torch import nn

class OcrModel(nn.Module):
    @abstractmethod
    def forward(self, features: list[torch.Tensor], labels: list[torch.Tensor]) -> torch.Tensor:
        """Evaluates the model on an instance of an OCR task

        Args:
            feature: A `[C, H, W]`-tensor representing the word image.
            label: A `[L, D]`-tensor representing a one-hot encoding of the
                label.
        Returns:
            A `[L, D]`-tensor representing the distribution of each character.
        """
        pass