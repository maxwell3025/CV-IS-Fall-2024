#!/bin/bash
#SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 01:00:00  # Job time limit
#SBATCH -o train_mamba_%j.out  # %j = job ID
#SBATCH --constraint sm_70  # %j = job ID

mkdir -p /work/pi_jaimedavila_umass_edu/<YOUR WORK FOLDER HERE>
cd /work/pi_jaimedavila_umass_edu/<YOUR WORK FOLDER HERE>
module load cuda/12.2.1

if ! test -d .venv; then
    module load python/3.11.0
    python3 -m venv .venv
    module unload python/3.11.0
    source ./.venv/bin/activate
else
    source ./.venv/bin/activate
    if ! python --version | grep "Python 3.11.0"; then
        echo ".venv currently uses an incorrect python version. Please delete .venv and rerun this script"
        exit 1
    fi
fi

python -m ensurepip --upgrade
python -m pip install --upgrade pip
p
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# INSTALL PACKAGES HERE
# python -m pip install wheel packaging setuptools
# python -m pip install causal-conv1d
# python -m pip install mamba-ssm
# python -m pip install wandb

export WANDB_API_KEY=<YOUR KEY HERE>

python train.py
