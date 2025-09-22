import torch

from tqdm import tqdm
from collections import defaultdict

from treemort.training.output_processing import process_model_output, prepare_pred_and_target


def validate_one_epoch(model, criterion, metrics, val_loader, conf, device):
    model.eval()
    val_loss = 0.0
    val_metrics = defaultdict(float)

    val_progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            logits = process_model_output(model, images, conf.model)
            _, _, h, w = labels.shape
            preds, targets, buffer = prepare_pred_and_target(logits, labels, (h, w))

            loss = criterion(preds, targets, buffer=buffer)
            batch_metrics = metrics(preds, targets, buffer=buffer)

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


def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[..., i:i+th, j:j+tw]