#!/bin/bash
#
# Install dependencies

cd $(dirname $0)/..

test -e ./.venv/bin/activate || exit 1

source ./.venv/bin/activate

python -m pip install -e .

deactivate
