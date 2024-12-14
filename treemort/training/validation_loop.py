import torch

from tqdm import tqdm
from collections import defaultdict

from treemort.training.output_processing import process_model_output


def validate_one_epoch(model, criterion, metrics, val_loader, conf, device):
    model.eval()
    val_loss = 0.0
    val_metrics = defaultdict(float)

    val_progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_progress_bar):
            images, labels = images.to(device), labels.to(device)

            logits = process_model_output(model, images, conf)

            loss = criterion(logits, labels)
            val_loss += loss.item()

            pred_probs = torch.sigmoid(logits)

            batch_metrics = metrics(pred_probs, labels)

            for key, value in batch_metrics.items():
                val_metrics[key] += value.item() if isinstance(value, torch.Tensor) else value

            val_progress_bar.set_postfix({
                "Val Loss": val_loss / (batch_idx + 1),
                "IOU": val_metrics.get("iou_segments", 0.0) / (batch_idx + 1),
                "F1": val_metrics.get("f_score_segments", 0.0) / (batch_idx + 1),
            })

    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    return val_loss, dict(val_metrics)