import torch
from torch import nn
from torch import optim
from torch import Tensor
from model import HippoSsmLayer
from model import DiagonalSsmLayer
from model import ComplexDiagonalSsmLayer
from model import S6Layer
from matplotlib import pyplot
import numpy
import math

def getDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.randn((batch_num, num_channels, signal_length))
        label = data[:, :, :-delay]
        label = torch.cat([torch.randn((batch_num, num_channels, delay)), label], 2)
        yield data, label

def getSparseDataIterator(num_channels, signal_length, batch_num, delay):
    while True:
        data = torch.rand((batch_num, num_channels, signal_length))
        label = data[:, :, :-delay]
        label = torch.cat([torch.rand((batch_num, num_channels, delay)), label], 2)
        yield (data < 0.05).float(), (label < 0.05).float()

def train(model, delay):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_iterator = getDataIterator(1, 128, 16, delay)
    sample_data, sample_label = torch.zeros((1, 1, 128)), torch.zeros((1, 1, 128))
    sample_data[0, 0, 0] = 1
    sample_label[0, 0, delay] = 1
    if type(model) == S6Layer:
        sample_data = sample_data.transpose(-1, -2)

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    mse_history = []
    for(i, (data, label)) in enumerate(data_iterator):
        model.zero_grad()
        if type(model) == S6Layer:
            data = data.transpose(-1, -2)
        prediction = model(data)
        if type(model) == S6Layer:
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
                
                # fig, (kernel_plot, history_plot) = pyplot.subplots(1, 2)
                # kernel_plot.plot(graph_data)
                # kernel_plot.set_title("kernel")

                # history_plot.plot(numpy.arange(0, 1000, 10), mse_history)
                # history_plot.set_title("error")

                # pyplot.show()
                return mse_history
            if i % 10 == 0:
                sample_prediction: Tensor = model(sample_data)
                if type(model) == S6Layer:
                    sample_prediction = sample_prediction.transpose(-1, -2)
                mse_history.append(math.log(error.item()))
                print(error.item())

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
    if classname == "DiagonalSsmLayer":
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)
    if classname == "ComplexDiagonalSsmLayer":
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)

state_dimensions = 64

complex_diagonal_model = ComplexDiagonalSsmLayer(1, state_dimensions // 2)
complex_diagonal_model.apply(initializeWeightsNaive)
s6_model = S6Layer(1, state_dimensions)
hippo_model = HippoSsmLayer(1, state_dimensions)
hippo_model.apply(initializeWeightsNaive)
diagonal_model = DiagonalSsmLayer(1, state_dimensions)
diagonal_model.apply(initializeWeightsNaive)

print('Training complex diagonal model')
complex_diagonal_mse = train(complex_diagonal_model, 32)
print("Final dt: ", torch.exp(diagonal_model.log_dt).item())

print('Training mamba model')
s6_mse = train(s6_model, 32)
print("Final dt: N/A", )
# _, (del_plot, sig_plot, pred_plot) = pyplot.subplots(1,3)
# del_plot.matshow(s6_model.previous_Delta[0].detach())
# sig_plot.matshow(s6_model.previous_sequence[0].detach())
# pred_plot.matshow(s6_model.previous_prediction[0].detach())
# pyplot.show()

print('Training hippo model')
hippo_mse = train(hippo_model, 32)
print("Final dt: ", torch.exp(hippo_model.log_dt).item())

print('Training diagonal model')
diagonal_mse = train(diagonal_model, 32)
print("Final dt: ", torch.exp(diagonal_model.log_dt).item())

# fig, (kernel_plot, history_plot) = pyplot.subplots(1, 2)
# kernel_plot.plot(graph_data)
# kernel_plot.set_title("kernel")

# history_plot.plot(numpy.arange(0, 1000, 10), s6_mse, label="s6")
# history_plot.plot(numpy.arange(0, 1000, 10), hippo_mse, label="hippo")
# history_plot.plot(numpy.arange(0, 1000, 10), diagonal_mse, label="diagonal")
# history_plot.set_title("Loss")

pyplot.plot(numpy.arange(0, 1000, 10), s6_mse, label="s6")
pyplot.plot(numpy.arange(0, 1000, 10), hippo_mse, label="hippo")
pyplot.plot(numpy.arange(0, 1000, 10), diagonal_mse, label="diagonal")
pyplot.plot(numpy.arange(0, 1000, 10), complex_diagonal_mse, label="complex diagonal")
pyplot.plot(numpy.arange(0, 1000, 10), numpy.full(100, 0), label="baseline")
pyplot.ylim((-0.5,1))
pyplot.title("Log Loss")
pyplot.legend()
pyplot.show()
