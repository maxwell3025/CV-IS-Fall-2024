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
# This script trains a model using the provided config

if [ -f "./train_cfg.sh" ]; then
    ./scripts/venv_setup_gpu.sh || exit 1
    ./scripts/install_gpu.sh || exit 1
    source ./.venv/bin/activate

    module load cuda/12.6

    python -m cfg_ops_mamba config/arithmetic_mamba_only.yaml

    deactivate
else
    echo "Please run this script from the same folder as train_cfg.sh"
fi