import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import training_config, dataset_config, MambaConfig
from data_generator import generate_dataset
from regular_languages import get_example_1
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Define model
mambaconfig = MambaConfig()
model = MambaLMHeadModel(mambaconfig, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

# Validation function
def validate(step, machine, start, enumeration):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        inputs, targets = generate_dataset(machine, start, enumeration, dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, num_last_tokens=1).logits
        total += targets.size(0) * targets.size(1)
        correct += (outputs.argmax(2) == targets).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Validation Accuracy: {accuracy:.2f}%')
        if step != -1:
            wandb.log({
                "step": step,
                "accuracy": accuracy
            })

# Training function
def train(machine, start, enumeration):
    """
    Train the model.
    """
    model.train()
    start_time = time.time()
    for step in range(training_config["num_steps"]):
        step_loss = 0
        correct = 0
        total = 0
        inputs, targets = generate_dataset(machine, start, enumeration, dataset_config, training_config)
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
        logger.info(f'Step [{step+1}/{training_config["num_steps"]}], Loss: {step_loss/training_config["batch_size"]:.4f}, Accuracy: {accuracy:.2f}%')
        if step % training_config["val_interval"] == 0:
            validate(step, machine, start, enumeration)

    end_time = time.time()
    logger.info(f'Training completed in: {(end_time - start_time)/60:.2f} minutes')

if __name__ == '__main__':
    wandb.login()
    run = wandb.init(
        project="dfa-ops-mamba",
    )
    machine, start, enumeration = get_example_1(dataset_config["length"])
    train(machine, start, enumeration)
    validate(-1, machine, start, enumeration)

