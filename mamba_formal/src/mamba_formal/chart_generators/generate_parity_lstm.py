import copy
import json
from matplotlib import pyplot
import numpy
from unique_names_generator import get_random_name

log_location = f"./validation_logs_parity_lstm.json"

with open(log_location) as log_file:
    data = json.load(log_file)

runs = dict()
for entry in data:
    entry_index = copy.deepcopy(entry)
    del entry_index["step"]
    del entry_index["accuracy"]
    del entry_index["validation_length"]
    key = json.dumps(entry_index)
    if key not in runs:
        runs[key] = []
    runs[key].append(entry)

for (name, run) in runs.items():
    drops = json.loads(name)["mamba_config"]["mamba_drop_prob"] == 0.5
    n_layers = json.loads(name)["mamba_config"]["n_layer"]
    iteration = json.loads(name)["iteration"]

    step_set = set()
    validation_length_set = set()
    for instance in run:
        step_set.add(instance["step"])
        validation_length_set.add(instance["validation_length"])
    step_list = list(step_set)
    validation_length_list = list(validation_length_set)
    step_list.sort()
    validation_length_list.sort()
    grid = numpy.full((len(step_list), len(validation_length_list)), -1)
    for instance in run:
        step_index = step_list.index(instance["step"])
        validation_length_index = validation_length_list.index(instance["validation_length"])
        grid[step_index, validation_length_index] = instance["accuracy"]

    fig, ax = pyplot.subplots()
    fig.set_size_inches(
        6.4,
        4.8
    )

    image = ax.matshow(grid, vmin=50, vmax=100)
    title = f"Performance of 2-layer LSTM with {n_layers - 2} Mamba Layers"
    if drops:
        title += "(DropPath)"
    ax.set_title(title)

    ax.set_xticks(
        ticks=list(range(0, len(validation_length_list)))[::5],
        labels=validation_length_list[::5],
    )
    ax.set_xlabel("Length of Validation String")

    ax.set_yticks(
        ticks=list(range(0, len(step_list)))[::5],
        labels=step_list[::5],
    )
    ax.set_ylabel("Training Step")

    cbar = fig.colorbar(image, orientation="horizontal")
    cbar.set_label("Accuracy as Percentage")

    figure_name = f"parity_lstm_{drops}_{n_layers}_{iteration}"
    pyplot.savefig(
        f"./charts/{figure_name}.png",
        dpi=500,
    )
    
    