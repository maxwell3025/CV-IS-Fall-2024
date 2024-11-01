import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_lstm import MambaLMHeadModelLstm
from config import iterate_sweep, DatasetConfig, TrainingConfig, MambaConfig
from data_generator import generate_dataset, generate_dataset_multi
import os
from unique_names_generator import get_random_name
from context_free_grammars import CFGSymbol, get_arithmetic_expr, a_or_bb, parity, get_arithmetic_expr_all, parity_all
import json
import sys

# We set up a global logger for debugging purposes.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def validate_recognizer(
    language: CFGSymbol,
    model: nn.Module,
    dataset_config: DatasetConfig,
    additional_params: dict[str, any],
):
    """Test the accuracy of a language recognizer model against multiple input
    string lengths.

    Args:
        language: The language to be validated against
        model: The model that we want to test
        dataset_config: The configuration for generating data
        additional_params: Additional JSON objects to add to the record
    Returns:
        A list of JSON-compatible objects representing the validation logs
        produced by this function
    """
    log_object = []
    model.eval()
    with torch.no_grad():
        for validation_length in training_config.val_lengths:
            old_positive = dataset_config.positive_rate
            dataset_config.positive_rate = 0.5
            correct = 0
            total = 0
            inputs, targets = generate_dataset(
                grammar=language,
                length=validation_length,
                randomize=False,
                batch_size=dataset_config.batch_size,
                one_hot=dataset_config.one_hot,
                positive_rate=dataset_config.positive_rate,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs: torch.Tensor = model(inputs, num_last_tokens=1).logits
            total += targets.size(0) * targets.size(1)
            correct += (outputs.argmax(2) == targets).sum().item()
            accuracy = 100 * correct / total
            log_object.append(dict(
                accuracy=accuracy,
                validation_length=validation_length,
                dataset_config=dataset_config.__dict__,
                **additional_params,
            ))
            dataset_config.positive_rate = old_positive
    return log_object

def train_recognizer(
    language: CFGSymbol,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    mamba_config: MambaConfig,
    model: MambaLMHeadModel,
):
    """Trains a model to recognize instances of a language.

    Args:
        language: A CFG Symbol representing the language that we will train the
            model to recognize.
        training_config: A TrainingConfig object containing the training-related
            hyperparameters like step size and number of steps. For more info,
            see TrainingConfig.
        dataset_config: A DatasetConfig object containing the hyperparameters
            that define our dataset.
        mamba_config: A MambaConfig object containing the hyperparameters
            defining our model architecture.
        model: The model that we want to train.

    Returns:
        A tuple (model, logs) where model is the model after training and logs
        is a list of JSON log entries produced during training.
    """
    log_object = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
    model.train()
    start_time = time.time()
    for step in range(training_config.num_steps):
        batch_length = dataset_config.training_length
        inputs, targets = generate_dataset(
            grammar=language,
            length=batch_length,
            randomize=dataset_config.randomize_training_length,
            batch_size=dataset_config.batch_size,
            one_hot=dataset_config.one_hot,
            positive_rate=dataset_config.positive_rate,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).logits
        loss = criterion(torch.transpose(outputs, 1, 2), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % training_config.val_interval == 0:
            log_object = log_object + validate_recognizer(
                language=language,
                model=model,
                dataset_config=dataset_config,
                additional_params=dict(
                    step=step,
                    training_config=training_config.__dict__,
                    mamba_config=mamba_config.__dict__,
                )
            )
            model.train()
        print(step)
    end_time = time.time()
    train_time_mins = (end_time - start_time)/60
    logger.info(
        f"Training instance completed in: {train_time_mins:.2f} minutes"
    )
    return model, log_object

def validate_multi(
    languages: list[CFGSymbol],
    model: nn.Module,
    dataset_config: DatasetConfig,
    additional_params: dict[str, any],
):
    """Test the ability for a model to distinguish multiple context free
    languages.

    Args:
        languages: A list of CFGSymbol instances that represents a set of
            languages
        model (nn.Module): A PyTorch model that can distinguish between
            instances of the different languages
        dataset_config: A DatasetConfig object that defines how data instances
            are generated
        additional_params: A dictionary of additional fields that should be
            appended to each row of logs

    Returns:
        A list of JSON-compatible dicts representing the performance of the
        model against various input data lengths
    """
    log_object = []
    model.eval()
    with torch.no_grad():
        for validation_length in training_config.val_lengths:
            old_positive = dataset_config.positive_rate
            dataset_config.positive_rate = 0.5
            correct = 0
            total = 0
            inputs, targets = generate_dataset_multi(
                grammars=languages,
                length=validation_length,
                randomize=False,
                batch_size=dataset_config.batch_size,
                one_hot=dataset_config.one_hot,
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs: torch.Tensor = model(inputs, num_last_tokens=1).logits
            total += targets.size(0) * targets.size(1)
            correct += (outputs.argmax(2) == targets).sum().item()
            accuracy = 100 * correct / total
            log_object.append(dict(
                accuracy=accuracy,
                validation_length=validation_length,
                dataset_config=dataset_config.__dict__,
                **additional_params,
            ))
            dataset_config.positive_rate = old_positive
    return log_object

def train_multi(
    languages: list[CFGSymbol],
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    mamba_config: MambaConfig,
    model: MambaLMHeadModel,
):
    """Trains a model to differentiate between multiple languages.

    Args:
        language: A list of CFGSymbol instances representing the languages that
            we will train the model to distinguish.
        training_config: A TrainingConfig object containing the training-related
            hyperparameters like step size and number of steps. For more info,
            see TrainingConfig.
        dataset_config: A DatasetConfig object containing the hyperparameters
            that define our dataset.
        mamba_config: A MambaConfig object containing the hyperparameters
            defining our model architecture.
        model: The model that we want to train.

    Returns:
        A tuple (model, logs) where model is the model after training and logs
        is a list of JSON log entries produced during training.
    """
    log_object = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
    model.train()
    start_time = time.time()
    for step in range(training_config.num_steps):
        batch_length = dataset_config.training_length
        inputs, targets = generate_dataset_multi(
            grammars=languages,
            length=batch_length,
            randomize=dataset_config.randomize_training_length,
            batch_size=dataset_config.batch_size,
            one_hot=dataset_config.one_hot,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).logits
        loss = criterion(torch.transpose(outputs, 1, 2), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % training_config.val_interval == 0:
            log_object = log_object + validate_multi(
                languages=languages,
                model=model,
                dataset_config=dataset_config,
                additional_params=dict(
                    step=step,
                    training_config=training_config.__dict__,
                    mamba_config=mamba_config.__dict__,
                )
            )
            model.train()
        print(step)
    end_time = time.time()
    train_time_mins = (end_time - start_time)/60
    logger.info(
        f"Training instance completed in: {train_time_mins:.2f} minutes"
    )
    return model, log_object

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m cfg_ops_mamba <path to config file>")
    config_uri = sys.argv[1]

    # For this test, we will use the parity language
    Even, Odd = parity_all()
    language_set=[Even, Odd]

    # We will put all of our logs and checkpoints into a subfolder in output,
    # where the subfolder name is a random adjective_noun name.
    # We will save the folder name in a variable.
    folder_name = f"./output/{get_random_name(separator="_", style="lowercase")}"
    os.makedirs(folder_name, exist_ok=True)
    logger.info(f"Saving to output to {folder_name}")
    validation_logs_object = []

    # Set the device that we will use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    for (training_config, dataset_config,
        mamba_config) in iterate_sweep(config_uri):

        model = MambaLMHeadModelLstm(mamba_config, device=device)

        logger.info(f"Training {dict(
            **dataset_config.__dict__,
            **training_config.__dict__,
            **mamba_config.__dict__
        )}")

        model, validation_logs_object = train_multi(
            languages=language_set,
            training_config=training_config,
            dataset_config=dataset_config,
            mamba_config=mamba_config,
            model=model,
        )

        validation_logs_object += validate_multi(
            languages=language_set,
            model=model,
            dataset_config=dataset_config,
            additional_params=dict(
                step=training_config.num_steps,
                training_config=training_config.__dict__,
                mamba_config=mamba_config.__dict__,
            )
        )

        torch.save(
            model.state_dict(),
            "{}/{}_{}_{}_{}".format(
                folder_name,
                dataset_config.training_length,
                mamba_config.d_model,
                dataset_config.randomize_training_length,
                mamba_config.n_layer,
            ),
        )
    json_logs_path = f"{folder_name}/validation_logs.json"
    with open(json_logs_path, "w") as validation_logs_json:
        json.dump(validation_logs_object, validation_logs_json)