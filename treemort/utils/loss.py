import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple


class TreeMortalityLoss(nn.Module):
    def __init__(self, mask_weight=1.0, centroid_weight=0.7, sdt_weight=0.5, boundary_weight=1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.centroid_weight = centroid_weight
        self.sdt_weight = sdt_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target, buffer=None):
        if buffer is None:
            buffer = torch.ones_like(target[:, 0:1], dtype=torch.bool)

        buffer = buffer.squeeze(1)  # -> [B,h,w]

        losses = []

        if pred.shape[1] >= 1:  # mask
            mask_loss = self._mask_loss(pred[:, 0], target[:, 0], buffer)
            losses.append(self.mask_weight * mask_loss)

        if pred.shape[1] >= 2:  # centroid
            centroid_loss = self._centroid_loss(pred[:, 1], target[:, 1], buffer)
            losses.append(self.centroid_weight * centroid_loss)

        if pred.shape[1] >= 3:  # sdt + boundary
            sdt_loss, boundary_loss = self._sdt_boundary_loss(pred[:, 2], target[:, 2], buffer)
            losses.append(self.sdt_weight * sdt_loss + self.boundary_weight * boundary_loss)

        return sum(losses)

    def _mask_loss(self, pred, target, buffer):
        valid = buffer.bool()
        pred = pred[valid]
        target = target[valid]

        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = 1 - (2 * (pred.sigmoid() * target).sum() + 1e-8) / (pred.sigmoid().sum() + target.sum() + 1e-8)
        return 0.5 * bce + 0.5 * dice

    def _centroid_loss(self, pred, target, buffer):
        valid_mask = (target > 0.01) & buffer.bool()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return F.mse_loss(pred[valid_mask], target[valid_mask])

    def _sdt_boundary_loss(self, pred, target, buffer):
        buffer = buffer.bool()
        sdt_mask = (target != -1) & buffer
        boundary_mask = (target == -1) & buffer

        sdt_loss = F.smooth_l1_loss(pred[sdt_mask], target[sdt_mask])

        if boundary_mask.sum() > 0:
            # boundary_loss = torch.exp(pred[boundary_mask] + 1).mean()
            boundary_loss = F.l1_loss(pred[boundary_mask], target[boundary_mask])
        else:
            boundary_loss = torch.tensor(0.0, device=pred.device)

        return sdt_loss, boundary_loss


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[List[float]] = None,
    dice_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    smooth: float = 1e-8,
) -> torch.Tensor:
    
    logits = pred[:, 0]

    buffer = target[:, 3]
    target = target[:, 0]

    if buffer is not None:
        # target_size = buffer.shape[-2:]
        # logits = center_crop(logits, target_size)
        # target = center_crop(target, target_size)
        logits = logits * buffer
        target = target * buffer

    if class_weights is not None:
        assert len(class_weights) == 2, "Class weights must be [background, foreground]"
        weights = torch.where(
            target > 0.5,
            torch.tensor(class_weights[1], device=target.device),
            torch.tensor(class_weights[0], device=target.device),
        )
    else:
        weights = torch.ones_like(target)

    if buffer is not None:
        weights = weights * buffer

    bce_loss = F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction='mean')

    pred = torch.sigmoid(logits)
    intersection = (pred * target * weights).sum()
    union = (pred * weights).sum() + (target * weights).sum()
    dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)

    focal_loss = focal_loss_fn(logits, target, focal_alpha, focal_gamma, weights, buffer, smooth)

    return dice_weight * dice_loss + (1 - dice_weight) * focal_loss + bce_loss


def focal_loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
    weights: torch.Tensor,
    buffer: Optional[torch.Tensor],
    smooth: float,
) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss

    if buffer is not None:
        focal_loss = focal_loss * buffer
        valid_pixels = buffer.sum() + smooth
        return (focal_loss * weights).sum() / valid_pixels
    return (focal_loss * weights).mean()


def center_crop(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    _, _, h, w = tensor.size()
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[..., i : i + th, j : j + tw]


def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits)
    if pred.shape[-2:] != target.shape[-2:]:
        raise RuntimeError(f"Dice size mismatch: {pred.shape} vs {target.shape}")

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2.0 * intersection + smooth) / (union + smooth)


def weighted_dice_loss(logits, target, buffer=None, class_weights=None, smooth=1e-8):
    if buffer is not None:
        logits = logits * buffer
        target = target * buffer

    if class_weights is not None:
        weights = target * class_weights[1] + (1 - target) * class_weights[0]
    else:
        weights = 1.0

    pred = torch.sigmoid(logits)
    intersection = (pred * target * weights).sum()
    union = (pred * weights).sum() + (target * weights).sum()

    return 1 - (2.0 * intersection + smooth) / (union + smooth)


def mse_loss(logits, target):
    if logits.shape[-2:] != target.shape[-2:]:
        raise RuntimeError(f"MSE size mismatch: {logits.shape} vs {target.shape}")
    return F.mse_loss(torch.sigmoid(logits), target.float())
