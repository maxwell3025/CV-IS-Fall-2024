import torch
import random
import numpy as np
import torch.nn.functional as F
from cfg_ops_mamba.context_free_grammars import CFGSymbol
from cfg_ops_mamba.config import DatasetConfig, TrainingConfig
import math

def regular_sample(
        grammar: CFGSymbol,
        positive_rate: float,
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
        sample = grammar.sample_random(length)
        numerical_sequence = []
        for i in range(len(sample)):
            numerical_sequence.append(grammar.enumeration[sample[i]])
        x = torch.tensor(numerical_sequence, dtype=torch.long)
        y = torch.tensor((1,), dtype=torch.long)
    else:
        sample = grammar.sample_random_inv(length)
        numerical_sequence = []
        for i in range(len(sample)):
            numerical_sequence.append(grammar.enumeration[sample[i]])
        x = torch.tensor(numerical_sequence, dtype=torch.long)
        y = torch.tensor((0,), dtype=torch.long)
    if one_hot: x = F.one_hot(x, len(grammar.enumeration)).float()
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
    grammar: CFGSymbol,
    length: int,
    randomize: bool,
    batch_size: int,
    one_hot: bool,
    positive_rate: float,
    ):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """

    max_length = length
    if randomize:
        length = random.randint(1, max_length)
        while grammar.sample_random(length) == None or grammar.sample_random_inv(length) == None:
            length = random.randint(1, max_length)
    x = []
    y = []
    for _ in range(batch_size):
        x_instance, y_instance = regular_sample(
            grammar=grammar,
            positive_rate=positive_rate,
            length=length,
            one_hot=one_hot
        )
        x.append(x_instance)
        y.append(y_instance)
    return torch.stack(x), torch.stack(y)

def regular_sample_multi(
        grammars: list[CFGSymbol],
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
    chosen_grammar_index = random.randrange(len(grammars))
    grammar = grammars[chosen_grammar_index]
    sample = grammar.sample_random(length)
    numerical_sequence = []
    for i in range(len(sample)):
        numerical_sequence.append(grammars[0].enumeration[sample[i]])
    x = torch.tensor(numerical_sequence, dtype=torch.long)
    y = torch.tensor((chosen_grammar_index,), dtype=torch.long)
    if one_hot: x = F.one_hot(x, len(grammars[0].enumeration)).float()
    return x, y

"""
Examples:
print(torch_copying_data(10, 5, 10, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
print(torch_copying_data(10, 5, 10, variable=True, variable_length=False, batch_shape=(), one_hot=False, reverse=False))
Outputs:
(tensor([2, 2, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([2, 2, 2, 4, 6])) # copying memory task
(tensor([0, 6, 0, 0, 0, 0, 0, 6, 7, 0, 7, 5, 0, 0, 0, 9, 9, 9, 9, 9]), tensor([6, 6, 7, 7, 5])) # selective copying task
"""
def generate_dataset_multi(
    grammars: list[CFGSymbol],
    length: int,
    randomize: bool,
    batch_size: int,
    one_hot: bool,
    ):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    dataset_config (dict): Configuration for the dataset
    training_config (dict): Configuration for the training

    Returns:
    tuple: Generated inputs and targets
    """

    max_length = length
    if randomize:
        resample = True
        while resample:
            length = random.randint(1, max_length)
            resample = False
            for grammar in grammars:
                if grammar.sample_random(length) == None:
                    resample = True
                    break
    x = []
    y = []
    for _ in range(batch_size):
        x_instance, y_instance = regular_sample_multi(
            grammars=grammars,
            length=length,
            one_hot=one_hot
        )
        x.append(x_instance)
        y.append(y_instance)
    return torch.stack(x), torch.cat(y)
