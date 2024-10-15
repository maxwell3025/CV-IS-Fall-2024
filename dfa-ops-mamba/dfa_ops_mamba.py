import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import sweep_config, training_config, dataset_config, MambaConfig
from data_generator import generate_dataset
from regular_languages import get_example_1
import wandb
import csv
import os
import random
import numpy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Define model
mambaconfig = MambaConfig()

# Setup local logging
os.makedirs("./.logs", exist_ok=True)
if(os.path.isfile(".logs/latest.csv")):
    log_creation_time = os.path.getctime(".logs/latest.csv")
    log_creation_time = time.ctime(log_creation_time)
    log_creation_time = time.strptime(log_creation_time)
    new_filename = f".logs/{time.strftime('%Y%m%d-%H%M%S', log_creation_time)}.csv"
    os.rename(".logs/latest.csv", new_filename)

validation_logs = csv.writer(open(f".logs/latest.csv", "a"))

# Validation function
def validate(step, machine, start, enumeration, model):
    model.eval()
    with torch.no_grad():
        for validation_length in sweep_config["validation_length"]:
            correct = 0
            total = 0
            inputs, targets = generate_dataset(
                machine=machine,
                start=start,
                enumeration=enumeration,
                length=validation_length,
                dataset_config=dataset_config,
                training_config=training_config
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, num_last_tokens=1).logits
            total += targets.size(0) * targets.size(1)
            correct += (outputs.argmax(2) == targets).sum().item()
            accuracy = 100 * correct / total
            logger.info(f'Validation Accuracy: {accuracy:.2f}%')
            if step != -1:
                validation_logs.writerow([
                    step,
                    accuracy,
                    validation_length,
                    dataset_config["training_length"],
                    mambaconfig.d_model,
                    dataset_config["randomize_training_length"]
                ])

# Training function
def train(machine, start, enumeration):
    model = MambaLMHeadModel(mambaconfig, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    """
    Train the model.
    """
    model.train()
    start_time = time.time()
    for step in range(training_config["num_steps"]):
        step_loss = 0
        correct = 0
        total = 0
        batch_length = dataset_config["training_length"]
        if dataset_config["randomize_training_length"]:
            batch_length = random.randint(1, dataset_config["training_length"])
        inputs, targets = generate_dataset(
            machine=machine,
            start=start,
            enumeration=enumeration,
            length=batch_length,
            dataset_config=dataset_config,
            training_config=training_config
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).logits
        # print(outputs.shape)
        # print(targets.shape)
        loss = criterion(torch.transpose(outputs, 1, 2), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(2) == targets).sum().item()
        accuracy = 100 * correct / total
        # logger.info(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%')
        if step % training_config["val_interval"] == 0:
            validate(step, machine, start, enumeration, model)

    end_time = time.time()
    logger.info(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')
    return model

if __name__ == '__main__':
    wandb.login()
    run = wandb.init(
        project="dfa-ops-mamba-gridsearch",
    )

    abs_max_length = numpy.max(
        sweep_config["training_length"] + sweep_config["validation_length"]
    )
    print(abs_max_length)
    machine, start, enumeration = get_example_1(abs_max_length)

    for training_length in sweep_config["training_length"]:
        dataset_config["training_length"] = training_length
        for d_model in sweep_config["d_model"]:
            mambaconfig.d_model = d_model
            for randomize_training_length in sweep_config["randomize_training_length"]:
                dataset_config["randomize_training_length"] = randomize_training_length
                model = train(
                    machine=machine,
                    start=start,
                    enumeration=enumeration,
                )
                validate(
                    training_config["num_steps"],
                    machine,
                    start,
                    enumeration,
                    model
                )

