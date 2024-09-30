from model import NaiveSsmLayer
import torch
from torch import nn
from matplotlib import pyplot
import numpy

layer = NaiveSsmLayer(2, 3, 2)

def initializeWeight(module: nn.Module):
    classname = module.__class__.__name__
    if classname == "NaiveSsmLayer":
        nn.init.normal_(module.A)
        nn.init.normal_(module.B)
        nn.init.normal_(module.C)
    print(classname)

layer.apply(initializeWeight)
layer.B = torch.nn.Parameter(torch.Tensor((
    (1, 0),
    (0, 1),
    (0, 0)
)))
layer.A = torch.nn.Parameter(torch.Tensor((
    (0, 1, 0),
    (-1, 0, 0),
    (0, 0, 1)
)))
layer.C = torch.nn.Parameter(torch.Tensor((
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0)
)))

signal = torch.ones((160, 2))
output = layer(signal, 0.1)
pyplot.plot(output[:, 0].detach(), output[:, 1].detach())
pyplot.show()
