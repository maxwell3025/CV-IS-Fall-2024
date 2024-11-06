#!/bin/bash
#
# Tries to set up a venv in the unity environemnt

cd "$(dirname "$0")"

if ! test -d ../.venv; then
    module load python/3.12.3
    python3 -m venv ../.venv
    module unload python/3.12.3
    source ../.venv/bin/activate
else
    source ../.venv/bin/activate
    if ! python --version | grep "Python 3.12.3"; then
        echo ".venv currently uses an incorrect python version. Please delete .venv and rerun this script"
        exit 1
    fi
    deactivate
fi
