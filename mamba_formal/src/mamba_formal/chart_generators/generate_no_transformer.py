from matplotlib import pyplot
import numpy
import os
import pandas
import sys

def indexArray(array: numpy.ndarray):
    values = numpy.unique(array)
    values.sort()
    array = numpy.searchsorted(values, array)
    return array, values

log_location = f"./parity_no_transformer.csv"

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

training_length_index = training_length_values.searchsorted(16)
d_model_index = d_model_values.searchsorted(32)
random_index = random_values.searchsorted(1)
n_layer_index = n_layer_values.searchsorted(5)

data_grid = data_tensor[:, :, training_length_index, d_model_index, random_index, n_layer_index]

figure, axes = pyplot.subplots()

axes.set_title(
    "Accuracy of 5-Layer Mamba on Parity"
)

image = axes.matshow(data_grid, vmin=25, vmax=100)

axes.set_xticks(
    ticks=list(range(0, len(validation_length_values)))[::5],
    labels=validation_length_values[::5],
)
axes.set_xlabel("Length of Validation String")

axes.set_yticks(
    ticks=list(range(0, len(step_values)))[::5],
    labels=step_values[::5],
)
axes.set_ylabel("Training Step")

cbar = figure.colorbar(image, orientation="horizontal")
cbar.set_label("Accuracy as Percentage")

figure_name = f"mamba_no_self_attention"
pyplot.savefig(
    f"./charts/{figure_name}.png",
    dpi=500,
)