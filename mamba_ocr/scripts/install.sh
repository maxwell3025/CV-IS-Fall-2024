#!/bin/bash
#
# Install dependencies

cd $(dirname $0)/..

test -e ../venv/bin/activate || exit 1

source ../venv/bin/activate

python -m pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
MAMBA_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE python -m pip install -e ../mamba-debugger --no-binary=mamba_ssm
python -m pip install -e .[client]

deactivate
