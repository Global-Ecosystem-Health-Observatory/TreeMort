import torch

from tqdm import tqdm
from collections import defaultdict

from treemort.training.output_processing import process_model_output


def train_one_epoch(model, optimizer, scheduler, criterion, metrics, train_loader, conf, device):
    model.train()
    train_loss = 0.0
    train_metrics = defaultdict(float)

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        buffer_mask = labels[:, 3, :, :].unsqueeze(1)  # [B,1,H,W]
        _, _, h, w = buffer_mask.shape
        
        optimizer.zero_grad()

        logits = model(images)  # [B,3,H,W]
        
        cropped_logits = torch.cat([
            center_crop(logits[:, 0:1, :, :], (h, w)),  # Mask
            center_crop(logits[:, 1:2, :, :], (h, w)),  # Centroid
            center_crop(logits[:, 2:3, :, :], (h, w))   # Hybrid
        ], dim=1)  # [B,3,h,w]

        cropped_labels = center_crop(labels, (h, w))  # [B,4,h,w]
        
        loss = criterion(cropped_logits, cropped_labels)
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        with torch.no_grad():
            pred_probs = torch.sigmoid(cropped_logits)
            batch_metrics = metrics(pred_probs, cropped_labels)

        train_loss += loss.item()
        for key, value in batch_metrics.items():
            train_metrics[key] += value.item() if torch.is_tensor(value) else value

        train_progress_bar.set_postfix({
            "Loss": f"{train_loss/(batch_idx+1):.4f}",
            "IOU": f"{train_metrics.get('iou_segments',0)/(batch_idx+1):.4f}",
            "F1": f"{train_metrics.get('f_score_segments',0)/(batch_idx+1):.4f}"
        })

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss, dict(train_metrics)


def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[..., i:i+th, j:j+tw]