import seaborn
import numpy
from matplotlib import pyplot
from matplotlib.widgets import Slider

def indexArray(array: numpy.ndarray):
    values = numpy.unique(array)
    values.sort()
    array = numpy.searchsorted(values, array)
    return array, values

data = numpy.genfromtxt(".logs/latest.csv", delimiter=",")

step_index, step_values                           = indexArray(data[:, 0])
validation_length_index, validation_length_values = indexArray(data[:, 2])
training_length_index, training_length_values     = indexArray(data[:, 3])
accuracy = data[:, 1]

data_tensor = numpy.zeros((
    step_values.size,
    validation_length_values.size,
    training_length_values.size
))

data_tensor[step_index, validation_length_index, training_length_index] = accuracy

figure = pyplot.figure()
axes = pyplot.axes()

training_length_slider = Slider(
    figure.add_axes([0.125, 0.0, 0.75, 0.03]),
    "Training Length",
    valmin=numpy.min(training_length_values),
    valmax=numpy.max(training_length_values),
    valinit=2,
    valstep=training_length_values)

def update(val):
    axes.set_title(
        "Accuracy v.s (Step, Validation Length)\n"
        f"Training Length = {training_length_slider.val}"
        )

    training_length_ind = numpy.where(training_length_values == training_length_slider.val)
    training_length_ind = numpy.array(training_length_ind).item()
    data_grid = data_tensor[:, :, training_length_ind]
    seaborn.heatmap(
        data_grid,
        vmin=0.0,
        vmax=100.0,
        xticklabels=validation_length_values,
        yticklabels=step_values,
        cbar=False,
        ax=axes,
    )
training_length_slider.on_changed(update)
update(2)
pyplot.show()