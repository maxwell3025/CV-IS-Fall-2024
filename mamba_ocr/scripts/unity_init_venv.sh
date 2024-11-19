#!/bin/bash
#
# Ensures that a venv exists and has the correct version.

cd $(dirname $0)/..

PYTHON_VERSION=3.12.3

source /etc/profile

if ! test -d ./.venv; then
    module load python/$PYTHON_VERSION
    python3 -m venv ./.venv
    module unload python/$PYTHON_VERSION
    source ./.venv/bin/activate
else
    source ./.venv/bin/activate
    if ! python --version | grep "Python $PYTHON_VERSION"; then
        echo ".venv currently uses an incorrect python version. Please delete .venv and rerun this script"
        exit 1
    fi
fi

python -m ensurepip --upgrade
python -m pip install --upgrade pip

deactivate