import torch
from torch import nn
from torch import optim
from torch import Tensor
from model import HippoSsmLayer
from model import DiagonalSsmLayer
from model import MambaSsmLayer
from matplotlib import pyplot
import numpy

def getDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.randn((batch_num, num_channels, signal_length))
        label = data[:, :, :-delay]
        label = torch.cat([torch.randn((batch_num, num_channels, delay)), label], 2)
        yield data, label

def getSparseDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.randn((batch_num, num_channels, signal_length))
        label = data[:, :, :-delay]
        label = torch.cat([torch.randn((batch_num, num_channels, delay)), label], 2)
        yield (data > 2).float(), (label > 2).float()

def train(model, delay):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_iterator = getSparseDataIterator(1, 128, 16, delay)
    sample_data, sample_label = torch.zeros((1, 1, 128)), torch.zeros((1, 1, 128))
    sample_data[0, 0, 0] = 1
    sample_label[0, 0, delay] = 1
    if type(model) == MambaSsmLayer:
        sample_data = sample_data.transpose(-1, -2)

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    mse_history = []
    for(i, (data, label)) in enumerate(data_iterator):
        model.zero_grad()
        if type(model) == MambaSsmLayer:
            data = data.transpose(-1, -2)
        prediction = model(data)
        if type(model) == MambaSsmLayer:
            prediction = prediction.transpose(-1, -2)
        error = criterion(prediction, label)
        # print(model.A[0])
        # print(error)
        error.backward()
        optimizer.step()
        with torch.no_grad():
            if i == 1000:
                graph_data = torch.cat(
                    [
                        sample_prediction.detach().reshape((128, 1)),
                        sample_label.detach().reshape((128, 1))
                    ]
                    ,1
                )
                
                fig, (kernel_plot, history_plot) = pyplot.subplots(1, 2)
                kernel_plot.plot(graph_data)
                kernel_plot.set_title("kernel")

                history_plot.plot(numpy.arange(0, 1000, 10), mse_history)
                history_plot.set_title("error")

                pyplot.show()
                break
            if i % 10 == 0:
                sample_prediction: Tensor = model(sample_data)
                if type(model) == MambaSsmLayer:
                    sample_prediction = sample_prediction.transpose(-1, -2)
                mse_history.append(error.item())
                print(criterion(sample_prediction, sample_label))

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

state_dimensions = 7
s6_model = MambaSsmLayer(1, state_dimensions)
hippo_model = HippoSsmLayer(1, state_dimensions)
hippo_model.apply(initializeWeightsNaive)
diagonal_model = DiagonalSsmLayer(1, state_dimensions)
diagonal_model.apply(initializeWeightsNaive)
print('Training mamba model')
train(s6_model, 32)
print('Training hippo model')
train(hippo_model, 32)
print('Training diagonal model')
train(diagonal_model, 32)