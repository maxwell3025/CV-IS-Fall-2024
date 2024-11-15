#!/bin/bash
#
# Tries to set up a venv in the unity environemnt

cd "$(dirname "$0")"

if ! test -d ../../mamba-language-ops-venv; then
    echo "Please create a venv for Python 3.12 in $(realpath ../../mamba-language-ops-venv)"
    exit 1
else
    source ../../mamba-language-ops-venv/bin/activate
    if ! python --version | grep "Python 3.12"; then
        echo "venv currently uses an incorrect python version. Please delete $(realpath ../../mamba-language-ops-venv) and rerun this script"
        exit 1
    fi
    deactivate
fi
