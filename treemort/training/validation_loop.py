import numpy as np
import torch
from tqdm import tqdm

from treemort.training.output_processing import process_model_output


def validate_one_epoch(model, criterion, metrics, val_loader, conf, device):
    model.eval()
    val_loss = 0.0
    val_metrics = {}
    notna_batch_counts = {}

    class_weights = torch.tensor(conf.class_weights, dtype=torch.float32).to(device)

    val_progress_bar = tqdm(val_loader, desc=f"Validation", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_progress_bar):
            images, labels = images.to(device), labels.to(device)

            logits = process_model_output(model, images, conf)
            
            loss = criterion(logits, labels, class_weights=class_weights)

            val_loss += loss.item()

            pred_probs = torch.sigmoid(logits)

            batch_metrics = metrics(pred_probs, labels)
            for key, value in batch_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                    notna_batch_counts[key] = 0
                if not value.isnan():
                    val_metrics[key] += value.item()
                    notna_batch_counts[key] += 1

            val_progress_bar.set_postfix({"Val Loss": val_loss / (batch_idx + 1)})

    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= notna_batch_counts[key] if notna_batch_counts[key] != 0 else np.nan

    return val_loss, val_metrics
