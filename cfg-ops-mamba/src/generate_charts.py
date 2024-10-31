import numpy
from matplotlib import pyplot
from matplotlib.widgets import Slider
import json
import sys
import os
import copy
from unique_names_generator import get_random_name

if len(sys.argv) != 2:
    print("Usage: python generate_charts.py <path to file>")
    exit(1)
log_location = sys.argv[1]
if not os.path.isfile(log_location):
    log_location = f"./output/{log_location}/validation_logs.json"

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
        12.8,
        19.2
    )
    ax.set_anchor("N")
    image = ax.matshow(grid)
    fig.colorbar(image)
    ax.set_title("String Length and Timestep vs Accuracy")
    ax.set_xticks(
        ticks=list(range(0, len(validation_length_list))),
        labels=validation_length_list,
    )
    ax.set_yticks(
        ticks=list(range(0, len(step_list))),
        labels=step_list,
    )
    formatted_json_text = json.dumps(json.loads(name))
    pyplot.text(
        0.4, 0.1,
        formatted_json_text,
        wrap=True,
        transform=fig.dpi_scale_trans,
    )
    # pyplot.show()
    os.makedirs(f"output/{sys.argv[1]}/charts/", exist_ok=True)
    pyplot.savefig(
        f"output/{sys.argv[1]}/charts/{get_random_name(separator="_", style="lowercase")}.png",
    )
    
    