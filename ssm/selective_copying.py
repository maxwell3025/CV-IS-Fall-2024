import random
import numpy
import torch
from torch import nn
from torch.nn import functional
from torch import optim
from model import S6Layer
from model import HippoSsmLayerTransposed
from matplotlib import pyplot

def getSelectiveCopyIterator(batch_count, avg_T):
    while True:
        input = []
        label = []
        for batch_num in range(batch_count):
            current_input = numpy.full(avg_T * 2, 8)
            current_output = numpy.full(avg_T * 2, 8)
            tokens = [random.randint(0, 7) for _ in range(10)]
            T = avg_T + random.randint(-10, 10)
            current_input[T] = 9
            current_output[T] = 9
            positions = random.sample(range(T), 10)
            positions.sort()
            for i in range(10):
                current_input[positions[i]] = tokens[i]
                current_output[positions[i]] = tokens[i]
                current_output[T + i + 1] = tokens[i]
            input.append(current_input[numpy.newaxis, :])
            label.append(current_output[numpy.newaxis, :])
        input = numpy.concatenate(input, 0)
        label = numpy.concatenate(label, 0)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        input = functional.one_hot(input).float()
        label = functional.one_hot(label).float()
        fig, (input_plot, label_plot) = pyplot.subplots(1, 2)
        input_plot.matshow(input[0].detach())
        label_plot.matshow(label[0].detach())
        pyplot.show()
        yield input, label

class MambaLayer(nn.Module):
    def __init__(self, input_shape) -> None:
        super(MambaLayer, self).__init__()
    def forward():
        pass

def train(model):
    data_iterator = getSelectiveCopyIterator(16, 40)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    for (i, (data, label)) in enumerate(data_iterator):
        model.zero_grad()
        prediction = model(data)
        error = criterion(prediction, label)
        error.backward()
        optimizer.step()
        if i % 10 == 0:
            print(error.item())
        if i == 1000:
            break

hippo = HippoSsmLayerTransposed(10, 100)
nn.init.normal_(hippo.layer.B)
nn.init.normal_(hippo.layer.C)
mamba = S6Layer(10, 100)
print("Mamba")
train(mamba)
print("Hippo")
train(hippo)
