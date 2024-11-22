#!/bin/bash
#
# Install dependencies

cd $(dirname $0)/..

test -e ./.venv/bin/activate || exit 1

source ./.venv/bin/activate

python -m pip install wheel packaging
python -m pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .[unity]


deactivate
