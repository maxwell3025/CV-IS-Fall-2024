from cfg_ops_mamba.config import iterate_sweep, DatasetConfig, TrainingConfig, MambaConfig
from cfg_ops_mamba.mamba_lstm import MambaLMHeadModelLstm
from cfg_ops_mamba.models import sequence_stack
import json
import logging
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import os
import sys
import synthetic_languages
import time
import torch
from torch import nn
from torch import optim
from unique_names_generator import get_random_name
from ..data_loaders import ocr_task

# We set up a global logger for debugging purposes.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def train(
    task: ocr_task.OcrTask,
    model: nn.Module,  
):
    optimizer = optim.adamw.AdamW()
    
