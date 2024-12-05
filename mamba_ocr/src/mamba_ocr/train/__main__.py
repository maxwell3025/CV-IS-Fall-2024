from .. import data_loaders
from .. import models
from . import config
import logging
import torch
from torch import optim
from torch import nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

optimizers = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    log_object = []
    # TODO grab this from command line args
    for conf in config.generate_cases("config/medmamba_synth_mnist_like.yaml"):
        dataset_type = data_loaders.datasets[conf["dataset_type"]]
        dataset: data_loaders.ocr_task_base.OcrTaskBase = dataset_type(
            **conf["dataset_config"]
        )

        dataset_train = data_loaders.partition_task.PartitionTask(dataset, (0, 0.9))
        dataset_test = data_loaders.partition_task.PartitionTask(dataset, (0.9, 0.95))
        dataset_val = data_loaders.partition_task.PartitionTask(dataset, (0.95, 1))

        model_type = models.models[conf["model_type"]]
        model: models.ocr_model.OcrModel = model_type(
            d_feature=dataset.d_feature,
            d_label=dataset.d_alphabet,
            **conf["model_config"]
        )
        model = model.to(device)

        optimizer_type = optimizers[conf["optimizer_type"]]
        optimizer: torch.optim.Optimizer = optimizer_type(
            params=model.parameters(),
            **conf["optimizer_config"]
        )

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(conf["train_config"]["epochs"]):
            logger.info(f"Began training epoch {epoch + 1}")

            # Train the model on the training dataset
            train_loss = 0
            n_train = 0
            model.train()
            for batch in dataset_train.batches(conf["train_config"]["batch_size"]):
                for features, labels in batch:
                    features = [feature.to(device) for feature in features]
                    labels = [label.to(device) for label in labels]
                    result = torch.cat(model(features, labels))
                    label = torch.cat(labels)
                    loss: torch.Tensor = criterion(result, torch.argmax(label, dim=1))
                    train_loss += loss.item()
                    n_train += 1
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

            # Evaluate the model
            # model.eval()
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch + 1}")
            logger.info("-" * 80)
            logger.info(f"Per-token train loss: {train_loss/n_train}")
            logger.info("-" * 80)
            total_tokens = 0
            total_loss = 0
            total_correct = 0
            sum_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            for batch in dataset_val.batches(conf["val_config"]["batch_size"]):
                for features, labels in batch:
                    features = [feature.to(device) for feature in features]
                    labels = [label.to(device) for label in labels]
                    result: torch.Tensor = torch.cat(model(features, labels))
                    total_tokens += result.shape[0]
                    total_loss += sum_criterion(result, torch.argmax(torch.cat(labels), dim=1)).item()
                    result_argmax = torch.argmax(result, dim=1)
                    result_one_hot = nn.functional.one_hot(result_argmax, dataset.d_alphabet)
                    total_correct += (result_one_hot * torch.cat(labels)).sum().item()
            logger.info(f"Per-token in-context loss: {total_loss / total_tokens}")
            logger.info(f"Per-token in-context accuracy: {total_correct / total_tokens}")
            logger.info("-" * 80)
            total_tokens = 0
            total_loss = 0
            total_correct = 0
            sum_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            for batch in dataset_val.batches_no_context(conf["val_config"]["batch_size"]):
                for features, labels in batch:
                    features = [feature.to(device) for feature in features]
                    labels = [label.to(device) for label in labels]
                    result: torch.Tensor = model(features, labels)[0]
                    total_tokens += result.shape[0]
                    total_loss += sum_criterion(result, torch.argmax(torch.cat(labels), dim=1)).item()
                    result_argmax = torch.argmax(result, dim=1)
                    result_one_hot = nn.functional.one_hot(result_argmax, dataset.d_alphabet)
                    total_correct += (result_one_hot * torch.cat(labels)).sum().item()
            logger.info(f"Per-token out-of-context loss: {total_loss / total_tokens}")
            logger.info(f"Per-token out-of-context accuracy: {total_correct / total_tokens}")
            logger.info("-" * 80)
            total_tokens = 0
            total_loss = 0
            total_correct = 0
            sum_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            for batch in dataset_val.batches_shuffled(conf["val_config"]["batch_size"]):
                for features, labels in batch:
                    features = [feature.to(device) for feature in features]
                    labels = [label.to(device) for label in labels]
                    result: torch.Tensor = torch.cat(model(features, labels))
                    total_tokens += result.shape[0]
                    total_loss += sum_criterion(result, torch.argmax(torch.cat(labels), dim=1)).item()
                    result_argmax = torch.argmax(result, dim=1)
                    result_one_hot = nn.functional.one_hot(result_argmax, dataset.d_alphabet)
                    total_correct += (result_one_hot * torch.cat(labels)).sum().item()
            logger.info(f"Per-token shuffled-context loss: {total_loss / total_tokens}")
            logger.info(f"Per-token shuffled-context accuracy: {total_correct / total_tokens}")
            logger.info("=" * 80)
            model.zero_grad()
