import torch
from torch import nn
from torch import optim
from torch import Tensor
from model import NaiveSsmLayer
from matplotlib import pyplot
import numpy

def getDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.randn((batch_num, signal_length, num_channels))
        label = data[:, :-delay, :]
        label = torch.cat([torch.randn((batch_num, delay, num_channels)), label], 1)
        yield data, label

def initializeWeight(module: nn.Module):
    classname = module.__class__.__name__
    if classname == "NaiveSsmLayer":
        with torch.no_grad():
            module.A.copy_(-torch.eye(module.state_dimensions))
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)

def train(model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.apply(initializeWeight)

    data_iterator = getDataIterator(1, 32, 16, 1)
    sample_data, sample_label = next(getDataIterator(1, 32, 1, 1))

    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    for(i, (data, label)) in enumerate(data_iterator):
        model.zero_grad()
        prediction = model(data, 0.1)
        error = criterion(prediction, label)
        # print(model.A[0])
        # print(error)
        error.backward()
        optimizer.step()
        if i % 1000 == 0:
            with torch.no_grad():
                sample_prediction: Tensor = model(sample_data, 0.1)
                
                print(criterion(sample_prediction, sample_label))
                # graph_data = torch.cat(
                #     [
                #     sample_prediction.detach().reshape((32, 1)) + 1,
                #     sample_label.detach().reshape((32, 1))
                #     ]
                #     ,1
                # )
                
                # pyplot.plot(graph_data)
                # pyplot.show()
            
in_dimensions = 1
state_dimensions = 16
out_dimensions = 1
model = NaiveSsmLayer(in_dimensions, state_dimensions, out_dimensions)
train(model)