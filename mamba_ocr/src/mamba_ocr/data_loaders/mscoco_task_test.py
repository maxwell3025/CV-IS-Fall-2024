from . import mscoco_task
from matplotlib import pyplot
import numpy
positional_encoding_vectors = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
task = mscoco_task.MsCocoTask("./data/mscoco-text", positional_encoding_vectors, 64, False, False, False, False)
for batch in task.batches(1):
    [feature1, feature2, *_], [label1, label2, *_] = batch[0]

    pyplot.imshow(feature1.permute(1, 2, 0))
    label1 = label1.argmax(dim=1)
    label1 = label1.tolist()
    label1 = [task.get_index_alphabet(x) for x in label1]
    print(label1)
    pyplot.show()

    pyplot.imshow(feature2.permute(1, 2, 0))
    label2 = label2.argmax(dim=1)
    label2 = label2.tolist()
    label2 = [task.get_index_alphabet(x) for x in label2]
    print(label2)
    pyplot.show()