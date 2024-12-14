import torch

from tqdm import tqdm
from collections import defaultdict
from treemort.training.output_processing import process_model_output


def train_one_epoch(model, optimizer, criterion, metrics, train_loader, conf, device):
    model.train()
    train_loss = 0.0
    train_metrics = defaultdict(float)

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = process_model_output(model, images, conf)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred_probs = torch.sigmoid(logits)
        batch_metrics = metrics(pred_probs, labels)

        for key, value in batch_metrics.items():
            train_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value

        train_progress_bar.set_postfix({
            "Train Loss": train_loss / (batch_idx + 1),
            "IOU": train_metrics.get("iou_segments", 0.0) / (batch_idx + 1),
            "F1": train_metrics.get("f_score_segments", 0.0) / (batch_idx + 1),
        })

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, dict(train_metrics)
