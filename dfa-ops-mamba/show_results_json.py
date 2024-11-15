import json
import sys
import os
from plotly import express

if len(sys.argv) != 2:
    print("Usage: python show_results.py <path to file>")
    exit(1)
log_location = sys.argv[1]
if not os.path.isfile(log_location):
    log_location = f"./output/{log_location}/validation_logs.json"

data = json.load(open(log_location))
step = [item["step"] for item in data]
acc = [item["accuracy"] for item in data]
val_len = [item["validation_length"] for item in data]
fig = express.scatter_3d(
    x=step,
    y=val_len,
    z=acc,
    labels=dict(
        x="Step",
        y="Validation Length",
        z="Accuracy",
    )
)
fig.update_traces(marker=dict(size=1))
fig.show()
