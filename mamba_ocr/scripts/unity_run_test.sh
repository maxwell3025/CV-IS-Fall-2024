#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o latest.log  # %j = job ID
#SBATCH --constraint sm_70
#SBATCH --mail-type=ALL
#
# This is the main run script for Unity

./scripts/unity_init_venv.sh || exit 1

./scripts/unity_install.sh || exit 1

source .venv/bin/activate

python -m mamba_ocr.train

deactivate
