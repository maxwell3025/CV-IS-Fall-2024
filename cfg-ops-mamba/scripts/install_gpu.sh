#!/bin/bash
cd "$(dirname "$0")"

test -e ../.venv/bin/activate || exit 1
source ../.venv/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip

python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install wheel packaging setuptools
python -m pip install causal-conv1d
python -m pip install mamba-ssm
python -m pip install unique-names-generator
python -m pip install pyyaml
