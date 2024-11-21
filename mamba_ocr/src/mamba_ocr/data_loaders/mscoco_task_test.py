from . import mscoco_task
from matplotlib import pyplot
import numpy
positional_encoding_vectors: numpy.ndarray = numpy.array([
    [1, 0],
    [0, 1],
    [1, 1],
])
task = mscoco_task.MsCocoTask("./data/mscoco-text", positional_encoding_vectors, (1, 1))
pyplot.matshow(task.features["train"][0][0].numpy().transpose())
pyplot.show()