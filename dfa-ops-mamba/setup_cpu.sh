if ! test -d .venv; then
    echo "Please create a .venv for Python 3.12"
else
    source ./.venv/bin/activate
    if ! python --version | grep "Python 3.12"; then
        echo ".venv currently uses an incorrect python version. Please delete .venv and rerun this script"
        exit 1
    fi
fi

python -m ensurepip --upgrade
python -m pip install --upgrade pip --quiet
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu --quiet
python -m pip install wheel packaging setuptools --quiet
# CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE python -m pip install causal-conv1d --quiet
# MAMBA_SKIP_CUDA_BUILD=TRUE python -m pip install mamba-ssm --quiet
MAMBA_SKIP_CUDA_BUILD=TRUE python -m pip install -e ../mamba-debugger/
python -m pip install wandb --quiet
python -m pip install unique-names-generator --quiet
# CPU only
python -m pip install numpy --quiet
python -m pip install pandas --quiet
python -m pip install matplotlib --quiet
python -m pip install pyqt6 --quiet
python -m pip install plotly --quiet
