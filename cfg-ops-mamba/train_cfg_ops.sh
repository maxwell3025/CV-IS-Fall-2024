#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o train_mamba_%j.out  # %j = job ID
#SBATCH --constraint sm_70
#SBATCH --mail-type=ALL


mkdir -p /work/pi_jaimedavila_umass_edu/maxwelltang_umass_edu/cfg-ops-mamba
cd /work/pi_jaimedavila_umass_edu/maxwelltang_umass_edu/cfg-ops-mamba
source .env
module load cuda/12.6

if ! test -d .venv; then
    module load python/3.12.3
    python3 -m venv .venv
    module unload python/3.12.3
    source ./.venv/bin/activate
else
    source ./.venv/bin/activate
    if ! python --version | grep "Python 3.12.3"; then
        echo ".venv currently uses an incorrect python version. Please delete .venv and rerun this script"
        exit 1
    fi
fi

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install wheel packaging setuptools
python -m pip install causal-conv1d
python -m pip install mamba-ssm
python -m pip install wandb
python -m pip install unique-names-generator
python -m pip install pyyaml

export WANDB_API_KEY

python cfg_ops_mamba.py
