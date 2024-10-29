import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import iterate_sweep, DatasetConfig, TrainingConfig, MambaConfig
from data_generator import generate_dataset
import os
import random
from unique_names_generator import get_random_name
from context_free_grammars import CFGSymbol, get_arithemtic_expr
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Validation function
def validate(
    step: int,
    language: CFGSymbol,
    model: nn.Module,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    mamba_config: MambaConfig,
    log_object: list[any],
):
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
                dataset_config=dataset_config,
                training_config=training_config
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs: torch.Tensor = model(inputs, num_last_tokens=1).logits
            total += targets.size(0) * targets.size(1)
            correct += (outputs.argmax(2) == targets).sum().item()
            accuracy = 100 * correct / total
            if step != -1:
                log_object.append({
                    "step": step,
                    "accuracy": accuracy,
                    "validation_length": validation_length,
                    "dataset_config": dataset_config.__dict__,
                    "training_config": training_config.__dict__,
                    "mamba_config": mamba_config.__dict__,
                })
            dataset_config.positive_rate = old_positive

# Training function
def train(
    language: CFGSymbol,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    log_object: list[any],
    model: MambaLMHeadModel,
):
    """
    Train the model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
    model.train()
    start_time = time.time()
    for step in range(training_config.num_steps):
        batch_length = dataset_config.training_length
        if dataset_config.randomize_training_length:
            batch_length = random.randint(1, dataset_config.training_length)
        inputs, targets = generate_dataset(
            grammar=language,
            length=batch_length,
            dataset_config=dataset_config,
            training_config=training_config
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).logits
        loss = criterion(torch.transpose(outputs, 1, 2), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % training_config.val_interval == 0:
            validate(
                step=step,
                language=language,
                model=model,
                training_config=training_config,
                dataset_config=dataset_config,
                mamba_config=mamba_config,
                log_object=log_object,
            )

    end_time = time.time()
    logger.info(f'Training instance completed in: {(end_time - start_time)/60:.2f} minutes')
    return model

if __name__ == '__main__':
    language = get_arithemtic_expr()

    # Setup local logging
    folder_name = get_random_name(separator="_", style="lowercase")
    os.makedirs(f"./output/{folder_name}", exist_ok=True)
    logger.info(f"Saving to output to ./output/{folder_name}")
    validation_logs_object = []

    for training_config, dataset_config, mamba_config in iterate_sweep("config/a_or_bb.yaml"):
        model = MambaLMHeadModel(mamba_config, device=device)

        logger.info(f"Training {dict(**dataset_config.__dict__, **training_config.__dict__, **mamba_config.__dict__)}")

        model = train(
            language=language,
            training_config=training_config,
            dataset_config=dataset_config,
            log_object=validation_logs_object,
            model=model,
        )
        validate(
            language=language,
            step=training_config.num_steps,
            model=model,
            log_object=validation_logs_object
        )

        torch.save(model.state_dict(), f"./output/{folder_name}/"
                   f"{dataset_config.training_length}_"
                   f"{mamba_config.d_model}_"
                   f"{dataset_config.randomize_training_length}_"
                   f"{mamba_config.n_layer}")
    json_logs_path = f"output/{folder_name}/validation_logs.json"
    validation_logs_json = open(json_logs_path, "w")
    json.dump(validation_logs_object, validation_logs_json)