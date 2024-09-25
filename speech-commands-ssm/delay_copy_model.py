import torch
from torch import nn
from torch.nn import functional
from torch import optim
from torch import Tensor
from model import NaiveSsmLayer
from model import HippoSsmLayer
from matplotlib import pyplot
import numpy

def getDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.randn((batch_num, num_channels, signal_length))
        label = data[:, :, :-delay]
        label = torch.cat([torch.randn((batch_num, num_channels, delay)), label], 2)
        yield data, label

def train(model, delay):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_iterator = getDataIterator(1, 32, 16, delay)
    sample_data, sample_label = torch.zeros((1, 1, 32)), torch.zeros((1, 1, 32))
    sample_data[0, 0, 0] = 1
    sample_label[0, 0, delay] = 1

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    for(i, (data, label)) in enumerate(data_iterator):
        model.zero_grad()
        prediction = model(data, 0.1)
        error = criterion(prediction, label)
        # print(model.A[0])
        # print(error)
        error.backward()
        optimizer.step()
        with torch.no_grad():
            if i % 100 == 0:
                sample_prediction: Tensor = model(sample_data, 0.1)
                print(criterion(sample_prediction, sample_label))
            if i % 10000 == 0:
                graph_data = torch.cat(
                    [
                    sample_prediction.detach().reshape((32, 1)),
                    sample_label.detach().reshape((32, 1))
                    ]
                    ,1
                )
                
                pyplot.plot(graph_data)
                pyplot.show()

def initializeWeightsNaive(module: nn.Module):
    """This is a better initialization
    """
    classname = module.__class__.__name__
    if classname == "NaiveSsmLayer":
        with torch.no_grad():
            new_A = torch.eye(module.state_dimensions)
            new_A = -new_A * torch.rand((module.state_dimensions, module.state_dimensions))
            module.A.copy_(new_A)
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)
    if classname == "HippoSsmLayer":
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)

state_dimensions = 16
model = HippoSsmLayer(1, state_dimensions)
model.apply(initializeWeightsNaive)
train(model, 8)