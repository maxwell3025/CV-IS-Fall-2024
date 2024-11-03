import torch
import random
import torch.nn.functional as F
from language_select_task import LanguageSelectTask

def regular_sample(
        task: LanguageSelectTask,
        distribution: list[float] | None,
        length: int,
        one_hot=False,
    ) -> torch.Tensor | None:
    """
    """
    if distribution == None:
        distribution = [
            1/task.language_count() for i in range(task.language_count())
        ]
    if sum(distribution) != 1:
        raise ValueError("distribution must sum to 1")
    if len(distribution) > task.language_count():
        raise ValueError("distribution length does not match the task")

    selection = random.random()
    selection_index = 0
    while True:
        selection -= distribution[selection_index]
        if(selection <= 0): break
        selection_index += 1

    sample = task.sample(length, selection_index)
    
    if sample == None: return None

    x = torch.tensor(sample, dtype=torch.long)
    y = torch.tensor((selection_index,), dtype=torch.long)
        
    if one_hot: x = F.one_hot(x, task.language_count()).float()
    return x, y

def generate_dataset(
    task: LanguageSelectTask,
    length: int,
    batch_size: int,
    randomize: bool,
    one_hot: bool,
    ):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    task: A LanguageSelectTask instance representing task to sample from.
    length: The maximum length for the instances.
    batch_size: The batch size to generate.
    randomize: A boolean representing whether the instance length should be
        randomized.
    one_hot: A boolean representing whether the instances should be converted to
        one-hot representation.

    Returns:
        A tuple (x, y) where x is a tuple of sentences and y is a tuple of
        labels
    """

    if randomize: length = random.randrange(length)

    x = []
    y = []
    for _ in range(batch_size):
        x_instance, y_instance = regular_sample(
            task=task,
            distribution=None,
            length=length,
            one_hot=one_hot,
        )
        x.append(x_instance)
        y.append(y_instance)
    if None in x: return None

    return torch.stack(x), torch.stack(y)
