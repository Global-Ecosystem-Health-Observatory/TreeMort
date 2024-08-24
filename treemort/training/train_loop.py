import torch

from tqdm import tqdm

from treemort.training.output_processing import process_model_output


def train_one_epoch(model, optimizer, criterion, metrics, train_loader, conf, device, image_processor, class_weights):
    model.train()
    train_loss = 0.0
    train_metrics = {}

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        predictions = process_model_output(model, images, conf, image_processor, labels, device)
        
        loss = criterion(predictions, labels, class_weights=class_weights)

        if conf.model in ["maskformer", "detr"]:
            loss.requires_grad = True

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        batch_metrics = metrics(predictions, labels)
        for key, value in batch_metrics.items():
            if key not in train_metrics:
                train_metrics[key] = 0.0
            train_metrics[key] += value.item()

        train_progress_bar.set_postfix({"Train Loss": train_loss / (batch_idx + 1)})

    # Average train loss and metrics
    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, train_metrics
