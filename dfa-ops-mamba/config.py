# Configuration for training
training_config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_steps": 4000,
    "val_interval": 100
}

# Configuration for dataset
dataset_config = {
    "n_tokens": 2,
    "length": 16,  # number of tokens to memorize
    "positive_rate": 0.5,
    "one_hot": False,
    "static": False,
}

# Configuration for Mamba model
class MambaConfig:
    d_model: int = 8
    d_intermediate: int = 0
    n_layer: int = 2
    vocab_size: int = dataset_config['n_tokens']
    ssm_cfg: dict = {}
    attn_layer_idx: list = []
    attn_cfg: dict = {}
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False

# class MambaConfig:

#     d_model: int = 2560
#     d_intermediate: int = 0
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = dict
#     attn_layer_idx: list = list
#     attn_cfg: dict = dict
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True