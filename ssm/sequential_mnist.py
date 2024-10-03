from datasets import load_dataset
import torch
from torch import nn
from torch import Tensor
from torch.utils import data
from torch import optim
from torchvision.transforms.functional import pil_to_tensor
from model import HippoSsmLayerTransposed
from matplotlib import pyplot
import math
import itertools

class SequentialMnistModel(nn.Module):
    def __init__(self) -> None:
        super(SequentialMnistModel, self).__init__()
        self.upscale = nn.Sequential(
            nn.Linear(1, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
        )
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.collapse = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10, bias=False),
            nn.Softmax(dim=-1)
        )
        # In the github, 256, 64 is the default initialization
        self.ssm_layer = HippoSsmLayerTransposed(32, 32)
        nn.init.normal_(self.ssm_layer.layer.B)
        nn.init.normal_(self.ssm_layer.layer.C)


    def forward(self, image: Tensor):
        L = image.shape[-1]

        x = image.unsqueeze(-1)

        x = self.upscale(x)

        x = x.transpose(-1, -2)
        x = self.batchnorm1(x)
        x = x.transpose(-1, -2)

        x = self.ssm_layer(x)

        x = self.collapse(x[..., L-1, :])
        return x

def train(model):
    dataset = load_dataset("ylecun/mnist")
    train_dataset = iter(dataset["train"])
    test_dataset = iter(dataset["test"])
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    for batch_num in range(1000):
        items = [next(train_dataset) for _ in range(4)]
        label = torch.tensor([item["label"] for item in items])
        images = [(pil_to_tensor(item["image"])/256).flatten().unsqueeze(0) for item in items]
        data = torch.cat(images, 0)
        # print(data.shape)

        model.zero_grad()
        prediction = model(data)
        error = criterion(prediction, label)
        error.backward()
        # print("error: ", error.item())
        optimizer.step()
        if batch_num % 10 == 0:
            print(error.item())
            print("label: ", label[0])
            print("prediction: ", prediction[0].detach().numpy())
        if batch_num == 1000:
            break
    
model = SequentialMnistModel()
train(model)
