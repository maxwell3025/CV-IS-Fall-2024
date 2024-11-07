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

# We set up a global logger for debugging purposes.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def validate(
    task: synthetic_languages.LanguageSelectTask,
    model: nn.Module,
    dataset_config: DatasetConfig,
    val_lengths: list[int],
    device: torch.device,
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
        val_lengths: A list of integers representing the string lengths that we
            will validate against.
        device: The device that we will use to run the validation step.

    Returns:
        A list of JSON-compatible dicts representing the performance of the
        model against various input data lengths
    """
    log_object = []
    model.eval()
    with torch.no_grad():
        for validation_length in val_lengths:
            old_positive = dataset_config.positive_rate
            dataset_config.positive_rate = 0.5
            inputs, targets = synthetic_languages.sample_batch(
                task=task,
                length=validation_length,
                batch_size=dataset_config.batch_size,
                randomize=False,
                one_hot=dataset_config.one_hot,
            )
            alphabet_length = inputs.shape[2]
            assert inputs.shape == (dataset_config.batch_size, validation_length, alphabet_length)
            assert targets.shape == (dataset_config.batch_size,)

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs: torch.Tensor = model(inputs, num_last_tokens=1).squeeze(dim=1)
            output_dim = outputs.shape[1]
            assert outputs.shape == (dataset_config.batch_size, output_dim)

            total = targets.shape[0]
            correct = (outputs.argmax(dim=1) == targets).sum().item()

            accuracy = 100 * correct / total
            print(accuracy)
            log_object.append(dict(
                accuracy=accuracy,
                validation_length=validation_length,
                dataset_config=dataset_config.__dict__,
                **additional_params,
            ))
            dataset_config.positive_rate = old_positive
    model.train()
    return log_object

def train(
    task: synthetic_languages.LanguageSelectTask,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    mamba_config: MambaConfig,
    model: MambaLMHeadModel,
    device: torch.device,
    iteration: int,
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
        device: The device that we will train on.

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
        inputs, targets = synthetic_languages.sample_batch(
            task=task,
            length=batch_length,
            batch_size=dataset_config.batch_size,
            randomize=dataset_config.randomize_training_length,
            one_hot=dataset_config.one_hot,
        )
        train_length_actual = inputs.shape[1]
        alphabet_length = inputs.shape[2]
        assert inputs.shape == (dataset_config.batch_size, train_length_actual, alphabet_length)
        assert targets.shape == (dataset_config.batch_size,)

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).squeeze(dim=1)
        output_dim = outputs.shape[1]
        assert outputs.shape == (dataset_config.batch_size, output_dim)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % training_config.val_interval == 0:
            log_object = log_object + validate(
                task=task,
                model=model,
                dataset_config=dataset_config,
                additional_params=dict(
                    step=step,
                    training_config=training_config.__dict__,
                    mamba_config=mamba_config.__dict__,
                    iteration=iteration,
                ),
                val_lengths=training_config.val_lengths,
                device=device,
            )
            model.train()
        print(step)
    end_time = time.time()
    train_time_mins = (end_time - start_time)/60
    logger.info(
        f"Training instance completed in: {train_time_mins:.2f} minutes"
    )
    return model, log_object

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m cfg_ops_mamba <path to config file>")
        exit(1)
    config_uri = sys.argv[1]

    # For this test, we will use the parity language
    task=synthetic_languages.a_or_bb_plus(64)

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
        mamba_config, iteration) in iterate_sweep(config_uri):

        model = sequence_stack.SequenceStack(mamba_config)
        model.to(device)

        logger.info(f"Training {dict(
            **dataset_config.__dict__,
            **training_config.__dict__,
            **mamba_config.__dict__
        )}")

        model, training_validation_logs = train(
            task=task,
            training_config=training_config,
            dataset_config=dataset_config,
            mamba_config=mamba_config,
            model=model,
            device=device,
            iteration=iteration,
        )

        validation_logs_object += training_validation_logs

        validation_logs_object += validate(
            task=task,
            model=model,
            dataset_config=dataset_config,
            additional_params=dict(
                step=training_config.num_steps,
                training_config=training_config.__dict__,
                mamba_config=mamba_config.__dict__,
                iteration=iteration,
            ),
            val_lengths=training_config.val_lengths,
            device=device,
        )

        torch.save(
            model.state_dict(),
            "{}/{}_{}_{}_{}".format(
                folder_name,
                dataset_config.training_length,
                mamba_config.d_intermediate,
                dataset_config.randomize_training_length,
                mamba_config.n_layer,
            ),
        )
    json_logs_path = f"{folder_name}/validation_logs.json"
    with open(json_logs_path, "w") as validation_logs_json:
        json.dump(validation_logs_object, validation_logs_json)

if __name__ == "__main__":
    main()
