import typing
import json
import yaml
import copy

# Configuration for training
class TrainingConfig:
    def __init__(self, config: dict[str, any]) -> None:
        self.batch_size:    int   = int  (config["batch_size"])
        self.learning_rate: float = float(config["learning_rate"])
        self.num_steps:     int   = int  (config["num_steps"])
        self.val_interval:  int   = int  (config["val_interval"])
        self.val_lengths:   list  = list (config["val_lengths"])

# Configuration for datasets
class DatasetConfig:
    def __init__(self, config: dict[str, any]) -> None:
        self.training_length:           int   = int  (config["training_length"])
        self.positive_rate:             float = float(config["positive_rate"])
        self.randomize_training_length: bool  = bool (config["randomize_training_length"])
        self.one_hot:                   bool  = bool (config["one_hot"])

# Configuration for Mamba model
class MambaConfig:
    def __init__(self, config: dict[str, any]) -> None:
        self.d_model:                 int  = int (config["d_model"])
        self.d_intermediate:          int  = int (config["d_intermediate"])
        self.n_layer:                 int  = int (config["n_layer"])
        self.vocab_size:              int  = int (config["vocab_size"])
        self.ssm_cfg:                 dict = dict(config["ssm_cfg"])
        self.attn_layer_idx:          list = list(config["attn_layer_idx"])
        self.attn_cfg:                dict = dict(config["attn_cfg"])
        self.rms_norm:                bool = bool(config["rms_norm"])
        self.residual_in_fp32:        bool = bool(config["residual_in_fp32"])
        self.fused_add_norm:          bool = bool(config["fused_add_norm"])
        self.pad_vocab_size_multiple: int  = int (config["pad_vocab_size_multiple"])
        self.tie_embeddings:          bool = bool(config["tie_embeddings"])

def from_dict(data: dict[str, any]):
    return TrainingConfig(data), DatasetConfig(data), MambaConfig(data)

def from_json(json_file: typing.IO):
    data = json.load(json_file)
    return from_dict(data)

sweep_config = {
    "training_length": [64],
    "d_model": [32],
    "n_layer": [5],
    "randomize_training_length": [True]
}

def iterate_sweep(filename: str):
    with open(filename, "r") as stream:
        data = yaml.safe_load(stream)
    base = data["base"]
    sweep_config = data["sweep"]
    total_cases = 1
    for sweep_dim in sweep_config:
        total_cases *= len(sweep_config[sweep_dim])
    for case_index in range(total_cases):
        for sweep_dim in sweep_config:
            sweep_choices = sweep_config[sweep_dim]
            sweep_choice_index = case_index % len(sweep_choices)
            base[sweep_dim] = sweep_choices[sweep_choice_index]
            case_index = case_index // len(sweep_choices)
        yield from_dict(base)
