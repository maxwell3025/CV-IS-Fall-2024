from datasets import load_dataset
import torch
from torch import nn
from torch.utils import data
from model import HippoSsmLayer

class SequentialMnistModel(nn.Module):
    def __init__(self) -> None:
        self.main = nn.Sequential(
            nn.Linear(1, 4),
            nn.Sigmoid(),
            nn.Linear(4, 16),
            nn.Sigmoid(),
            HippoSsmLayer(16, 16),
        )

def getDataLoader():
    dataset = load_dataset("ylecun/mnist")
    print(next(iter(dataset["train"])))
    data.DataLoader(
        dataset, batch_size=128, num_workers=2,
    )

def train():
    data = getDataLoader()
    
model = 
