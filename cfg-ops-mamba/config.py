# Configuration for training
training_config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_steps": 10000,
    "val_interval": 250
}

# training_config = {
#     "batch_size": 64,
#     "learning_rate": 0.0001,
#     "num_steps": 1000,
#     "val_interval": 100
# }

# Configuration for dataset
dataset_config = {
    "n_tokens": 9,
    "training_length": 16,
    "positive_rate": 0.5,
    "randomize_training_length": True,
    "one_hot": False,
    "static": False,
}

# Configuration for Mamba model
class MambaConfig:
    d_model: int = 8
    d_intermediate: int = 0
    n_layer: int = 12
    vocab_size: int = dataset_config['n_tokens']
    ssm_cfg: dict = {}
    attn_layer_idx: list = []
    attn_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False

sweep_config = {
    "training_length": [64],
    "validation_length": [i for i in range(1, 65)],
    "d_model": [16, 32],
    "n_layer": [2],
    "randomize_training_length": [True]
}