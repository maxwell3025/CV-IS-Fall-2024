# MAMBA OCR
This is the main project of this independent study.

## Requirements

This project assumes that you have access to the
[Unity cluster](https://unity.rc.umass.edu/).
The scripts and tooling are also designed for unix-based distros like Mac or
Linux.

## Usage
Setup your `.env` according to `.env.example`

To train an instance, run
```bash
./scripts/run_test.sh
```

After training is complete, pull training logs with
```bash
./scripts/pull.sh
```

## Folder Structure
- `config`: The folder where all of the training configurations are stored.
- `data`: The folder where datasets are stored.
- `scripts`:\
    This contains all of the utility scripts used by this project.
    Any scripts prefixed with `unity_` should only be run on the Unity cluster,
    and any scripts not prefixed with `unity_` should only be run on the local
    dev machine.
- `src/mamba_ocr`:\
    This contains all of the source code for our project.
- `src/mamba_ocr/charts`:\
    This contains the scripts for generating figures in the project.
- `src/mamba_ocr/data_loaders`:\
    This contains the different data loaders/generators used in our experiments.
- `src/mamba_ocr/models`:\
    This contains the different models/layers used in our project.
- `src/mamba_ocr/train`:\
    This contains the training script for our project.

## Configuration

Add new run configurations in `config` based on the existing configs.

Then, change line 23 of `scripts/unity_run_test.sh` to target your new config.
