import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import sweep_config, training_config, dataset_config, MambaConfig
from data_generator import generate_dataset
import wandb
import csv
import os
import random
import numpy
from unique_names_generator import get_random_name
from context_free_grammars import CFGSymbol, get_json_cfg

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Define model
mambaconfig = MambaConfig()

# Setup local logging
folder_name = get_random_name(separator="_", style="lowercase")
os.makedirs(f"./output/{folder_name}", exist_ok=True)
logger.info(f"Saving to output to ./output/{folder_name}")

validation_logs = csv.writer(open(f"output/{folder_name}/validation_logs.csv", "a"))

# Validation function
def validate(step, grammar: CFGSymbol, model):
    model.eval()
    with torch.no_grad():
        for validation_length in sweep_config["validation_length"]:
            old_positive = dataset_config["positive_rate"]
            dataset_config["positive_rate"] = 0.5
            correct = 0
            total = 0
            inputs, targets = generate_dataset(
                grammar=grammar,
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
            if step != -1:
                validation_logs.writerow([
                    step,
                    accuracy,
                    validation_length,
                    dataset_config["training_length"],
                    mambaconfig.d_model,
                    dataset_config["randomize_training_length"],
                    mambaconfig.n_layer,
                ])
            dataset_config["positive_rate"] = old_positive

# Training function
def train(grammar: CFGSymbol):
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
            batch_length = random.randint(18, dataset_config["training_length"])
        inputs, targets = generate_dataset(
            grammar=grammar,
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
            validate(step=step, grammar=grammar, model=model)

    end_time = time.time()
    logger.info(f'Training instance completed in: {(end_time - start_time)/60:.2f} minutes')
    return model

if __name__ == '__main__':
    wandb.login()
    run = wandb.init(
        project="dfa-ops-mamba-gridsearch",
    )

    abs_max_length = numpy.max(
        sweep_config["training_length"] + sweep_config["validation_length"]
    )

    grammar = get_json_cfg()

    for training_length in sweep_config["training_length"]:
        dataset_config["training_length"] = training_length
        for d_model in sweep_config["d_model"]:
            mambaconfig.d_model = d_model
            for randomize_training_length in sweep_config["randomize_training_length"]:
                dataset_config["randomize_training_length"] = randomize_training_length
                for n_layer in sweep_config["n_layer"]:
                    mambaconfig.n_layer=n_layer
                    logger.info(f"Training {dict(dataset_config, **training_config, **mambaconfig.__dict__)}")
                    model = train(
                        grammar=grammar
                    )
                    validate(
                        training_config["num_steps"],
                        grammar=grammar,
                        model=model
                    )
                    torch.save(model.state_dict(), f"./output/{folder_name}/{training_length}_{d_model}_{randomize_training_length}_{n_layer}")
