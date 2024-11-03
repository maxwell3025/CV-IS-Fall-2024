#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o train_mamba_%j.out  # %j = job ID
#SBATCH --constraint sm_70
#SBATCH --mail-type=ALL

cd "$(dirname "$0")"

./scripts/venv_setup_gpu.sh || exit 1
./scripts/install_gpu.sh || exit 1
./.venv/bin/activate

module load cuda/12.6

python -m cfg_ops_mamba ./config/parity_lstm.yaml
