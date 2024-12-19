# Formal Language Testing With Mamba
This repository contains the code for Maxwell's tests on formal languages.
To make imports cleaner, this project is formatted as a pyproject.

## File Structure
- `config`: The folder where all of the training configurations are stored.
- `scripts`: The folder where non-user Bash scripts are stored.
- `output`: The folder where all training logs will be placed.
- `src/mamba_formal`: The folder containing all of the Python code for this\
    module.
- `src/mamba_formal/chart_generators`: The folder containing all of the scripts\
    used to generate the charts in the paper
- `src/mamba_formal/models`: The folder containing all of the Python code for\
    the custom PyTorch modules used by this project.
- `src/mamba_formal/__main__.py`: The training script and entry point for the\
    module.
- `pull_unity.sh`: The script for downloading logs from the Unity cluster.
- `push_unity.sh`: The script for uploading local code to the Unity cluster.
- `train_cfg.sh`: The batch script for training.

## Requirements

This project assumes that you have access to the
[Unity cluster](https://unity.rc.umass.edu/).
The scripts and tooling are also designed for unix-based distros like Mac or
Linux.

## Usage

Configure your `.env` in accordance with `.env.example`

Use the following command to upload your local repo to Unity.

```bash
./push_unity.sh
```

Login to Unity, and go to your work directory:
```bash
cd /work/pi_<name of your PI>/<your username>/mamba_formal
```

Run the script
```bash
sbatch ./train_cfg.sh
```
## Configuration

Add new run configurations in `config` based on the existing configs.

Then, change line 20 of `train_cfg.sh` to target your new config.
