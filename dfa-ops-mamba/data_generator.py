import torch
import random
import numpy as np
import torch.nn.functional as F
from regular_languages import DfaState, sampleRandom, sampleRandomInv

def regular_sample(
        machine: set[DfaState],
        start: DfaState,
        positive_rate: float,
        enumeration: dict[str, int],
        length: int,
        one_hot=False,
    ):
    """
    Generate a dataset for a sequence copying task.
    This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
    https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

    Parameters:
    variable (bool): If True, selective copying task
    variable_length (bool): If True, randomize number of tokens to memorize
    batch_shape (tuple): Shape of the batch
    one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor

    Returns:
    tuple: Generated input sequence and target sequence
    """
    x = None
    y = None
    if random.random() < positive_rate:
        sample = sampleRandom(machine, start, length)
        numerical_sequence = []
        for i in range(len(sample)):
            numerical_sequence.append(enumeration[sample[i]])
        x = torch.tensor(numerical_sequence, dtype=torch.long)
        y = torch.tensor((1,), dtype=torch.long)
    else:
        sample = sampleRandomInv(machine, start, length)
        numerical_sequence = []
        for i in range(len(sample)):
            numerical_sequence.append(enumeration[sample[i]])
        x = torch.tensor(numerical_sequence, dtype=torch.long)
        y = torch.tensor((0,), dtype=torch.long)
        
    if one_hot: x = F.one_hot(x, len(enumeration)).float()
    return x, y

"""
Examples:
print(torch_copying_data(10, 5, 10, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
print(torch_copying_data(10, 5, 10, variable=True, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
Outputs:
(tensor([2, 2, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([2, 2, 2, 4, 6])) # copying memory task
(tensor([0, 6, 0, 0, 0, 0, 0, 6, 7, 0, 7, 5, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # selective copying task
"""
def generate_dataset(
    machine: set[DfaState],
    start: DfaState,
    enumeration: dict[str, int],
    dataset_config,
    training_config
    ):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """


    x = []
    y = []
    for _ in range(training_config["batch_size"]):
        x_instance, y_instance = regular_sample(
            machine,
            start,
            positive_rate=dataset_config["positive_rate"],
            enumeration=enumeration,
            length=dataset_config["length"],
            one_hot=dataset_config["one_hot"]
        )
        x.append(x_instance)
        y.append(y_instance)
    return torch.stack(x), torch.stack(y)


