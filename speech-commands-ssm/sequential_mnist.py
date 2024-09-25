from datasets import load_dataset
import torch
from torch import nn
from torch.utils import data

class SequentialMnistModel(nn.Module):
    def __init__(self) -> None:
        self.main = nn.Sequential([
            
        ])
def getDataLoader():
    dataset = load_dataset("ylecun/mnist")
    print(next(iter(dataset["train"])))
    data.DataLoader(
        dataset, batch_size=128, num_workers=2,
    )
getDataLoader()
