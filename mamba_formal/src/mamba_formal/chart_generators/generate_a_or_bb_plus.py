import copy
import json
from matplotlib import pyplot
import numpy
from unique_names_generator import get_random_name

log_location = f"./validation_logs_a_or_bb.json"

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
    is_positive_only = json.loads(name)["dataset_config"]["positive_rate"] == 1

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
    ax.set_title(
        "Performance of Multi-Layer Mamba Stack on (a|bb)+ (Positive Cases Only)"
        if is_positive_only else
        "Performance of Multi-Layer Mamba Stack on (a|bb)+")

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

    figure_name = (
        "a_or_bb_plus_positive_only"
        if is_positive_only else
        "a_or_bb_plus_mixed"
    )
    pyplot.savefig(
        f"{figure_name}.png",
        dpi=500,
    )
    
    