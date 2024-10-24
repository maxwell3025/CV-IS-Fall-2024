import seaborn
import numpy
from matplotlib import pyplot
from matplotlib.widgets import Slider
import pandas
import sys
import os

def indexArray(array: numpy.ndarray):
    values = numpy.unique(array)
    values.sort()
    array = numpy.searchsorted(values, array)
    return array, values

if len(sys.argv) != 2:
    print("Usage: python show_results.py <path to file>")
    exit(1)
log_location = sys.argv[1]
if not os.path.isfile(log_location):
    log_location = f"./output/{log_location}/validation_logs.csv"

data = pandas.read_csv(log_location, header=None)

step_indexes, step_values                           = indexArray(data.iloc[:, 0].to_numpy())
validation_length_indexes, validation_length_values = indexArray(data.iloc[:, 2].to_numpy())
training_length_indexes, training_length_values     = indexArray(data.iloc[:, 3].to_numpy())
d_model_indexes, d_model_values                     = indexArray(data.iloc[:, 4].to_numpy())
random_indexes, random_values                       = indexArray(data.iloc[:, 5].to_numpy())
random_values = random_values.astype(int)
n_layer_indexes, n_layer_values                     = indexArray(data.iloc[:, 6].to_numpy())

accuracy = data.iloc[:, 1].to_numpy()

data_tensor = numpy.zeros((
    step_values.size,
    validation_length_values.size,
    training_length_values.size,
    d_model_values.size,
    random_values.size,
    n_layer_values.size,
))

data_tensor[
    step_indexes,
    validation_length_indexes,
    training_length_indexes,
    d_model_indexes,
    random_indexes,
    n_layer_indexes
] = accuracy

figure = pyplot.figure()
axes = pyplot.axes([0.2, 0.1, 0.6, 0.8])

training_length_slider = Slider(
    figure.add_axes(
        [0.01, 0.1, 0.01, 0.8]
    ),
    "Training Length",
    valmin=numpy.min(training_length_values),
    valmax=numpy.max(training_length_values),
    valinit=2,
    valstep=training_length_values,
    orientation="vertical",
)

d_model_slider = Slider(
    figure.add_axes(
        [0.06, 0.1, 0.01, 0.8]
    ),
    "Model Dimension",
    valmin=numpy.min(d_model_values),
    valmax=numpy.max(d_model_values),
    valinit=2,
    valstep=d_model_values,
    orientation="vertical",
)

random_slider = Slider(
    figure.add_axes(
        [0.11, 0.1, 0.01, 0.8]
    ),
    "Randomize",
    valmin=numpy.min(random_values),
    valmax=numpy.max(random_values),
    valinit=2,
    valstep=random_values,
    orientation="vertical",
)

n_layer_slider = Slider(
    figure.add_axes(
        [0.16, 0.1, 0.01, 0.8]
    ),
    "Num Layers",
    valmin=numpy.min(n_layer_values),
    valmax=numpy.max(n_layer_values),
    valinit=2,
    valstep=n_layer_values,
    orientation="vertical",
)

def update(val=None):
    axes.set_title(
        "Accuracy v.s (Step, Validation Length)\n"
        f"Training Length = {training_length_slider.val}\n"
        f"Model Dimension = {d_model_slider.val}\n"
        f"Random = {random_slider.val}\n"
        f"Number of Layers = {n_layer_slider.val}\n"
    )

    training_length_index = training_length_values.searchsorted(training_length_slider.val)
    d_model_index = d_model_values.searchsorted(d_model_slider.val)
    random_index = random_values.searchsorted(random_slider.val)
    n_layer_index = n_layer_values.searchsorted(n_layer_slider.val)

    data_grid = data_tensor[:, :, training_length_index, d_model_index, random_index, n_layer_index]
    axes.matshow(
        data_grid,
    )
    axes.set_xticklabels(validation_length_values)
    axes.set_xticks([i for i in range(len(validation_length_values))])
    axes.set_yticklabels(step_values)
    axes.set_yticks([i for i in range(len(step_values))])
    # seaborn.heatmap(
    #     data_grid,
    #     vmin=0.0,
    #     vmax=100.0,
    #     xticklabels=validation_length_values,
    #     yticklabels=step_values,
    #     cbar=False,
    #     ax=axes,
    # )

training_length_slider.on_changed(update)
d_model_slider.on_changed(update)
random_slider.on_changed(update)
n_layer_slider.on_changed(update)
update()
pyplot.show()