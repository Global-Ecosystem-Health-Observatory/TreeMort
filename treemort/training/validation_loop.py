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
            
            buffer_mask = labels[:, 3, :, :].unsqueeze(1)  # [B, 1, H, W]
            _, _, h, w = buffer_mask.shape

            logits = process_model_output(model, images, conf.model)
            
            target_mask = labels[:, 0, :, :].unsqueeze(1)  # [B, 1, h, w]
            target_centroid = labels[:, 1, :, :].unsqueeze(1)  # [B, 1, h, w]
            target_hybrid = labels[:, 2, :, :].unsqueeze(1)  # [B, 1, h, w]
            targets = torch.cat([
                target_mask,        # Channel 0
                target_centroid,    # Channel 1
                target_hybrid,      # Channel 2
                buffer_mask         # Channel 3
            ], dim=1)  # [B, 4, h, w]

            loss = criterion(logits, targets)
            
            batch_metrics = metrics(logits, targets)

            val_loss += loss.item()
            for key, value in batch_metrics.items():
                val_metrics[key] += value.item() if torch.is_tensor(value) else value

            val_progress_bar.set_postfix({
                "Loss": f"{val_loss/(batch_idx+1):.4f}",
                "IOU": f"{val_metrics.get('iou_segments',0)/(batch_idx+1):.4f}",
                "F1": f"{val_metrics.get('f_score_segments',0)/(batch_idx+1):.4f}"
            })

    val_loss /= len(val_loader)
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)

    return val_loss, dict(val_metrics)