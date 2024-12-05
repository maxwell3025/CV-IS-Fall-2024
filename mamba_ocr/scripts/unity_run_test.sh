#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=16384  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o latest-5.log  # %j = job ID
#SBATCH --constraint sm_70
#SBATCH --mail-type=ALL
#
# Runs an experiment on Unity

source /etc/profile

module load cuda/12.6

./scripts/unity_init_venv.sh || exit 1

./scripts/unity_install.sh || exit 1

source .venv/bin/activate

python -m mamba_ocr.train config/medmamba_mscoco.yaml

deactivate
