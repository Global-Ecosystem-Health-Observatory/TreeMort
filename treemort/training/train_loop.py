import torch
import torch.distributed as dist

from tqdm import tqdm
from collections import defaultdict

from treemort.training.output_processing import process_model_output, prepare_pred_and_target


def train_one_epoch(model, optimizer, scheduler, criterion, metrics, train_loader, conf, device, is_main=True, is_distributed=False):
    model.train()
    train_loss = 0.0
    train_metrics = defaultdict(float)

    train_progress_bar = tqdm(train_loader, desc="Training", unit="batch", disable=not is_main)

    for batch_idx, (images, labels) in enumerate(train_progress_bar):
        images, labels = images.to(device), labels.to(device)

        buffer_mask = labels[:, 3, :, :].unsqueeze(1)  # [B,1,H,W]
        _, _, h, w = buffer_mask.shape

        optimizer.zero_grad()

        logits = process_model_output(model, images, conf.model)
        _, _, h, w = labels.shape
        preds, targets, buffer = prepare_pred_and_target(logits, labels, (h, w))

        loss = criterion(preds, targets, buffer=buffer)
        loss.backward()

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            batch_metrics = metrics(preds, targets, buffer=buffer)

        train_loss += loss.item()
        for key, value in batch_metrics.items():
            train_metrics[key] += value.item() if torch.is_tensor(value) else value

        if is_main:
            train_progress_bar.set_postfix({
                "Loss": f"{train_loss/(batch_idx+1):.4f}",
                "IOU": f"{train_metrics.get('iou_segments',0)/(batch_idx+1):.4f}",
                "F1": f"{train_metrics.get('f_score_segments',0)/(batch_idx+1):.4f}"
            })

    train_loss /= len(train_loader)
    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    if is_distributed:
        world_size = dist.get_world_size()
        loss_t = torch.tensor(train_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        train_loss = loss_t.item() / world_size
        for key in train_metrics:
            m_t = torch.tensor(train_metrics[key], device=device)
            dist.all_reduce(m_t, op=dist.ReduceOp.SUM)
            train_metrics[key] = m_t.item() / world_size

    return train_loss, dict(train_metrics)
