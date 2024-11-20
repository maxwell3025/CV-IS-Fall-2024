from . import mscoco_task
from matplotlib import pyplot
task = mscoco_task.MsCocoTask("./data/mscoco-text")
pyplot.matshow(task.sequences[0].numpy().transpose())
pyplot.show()