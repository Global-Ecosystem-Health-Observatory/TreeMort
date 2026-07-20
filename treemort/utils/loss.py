import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple


class TreeMortalityLoss(nn.Module):
    def __init__(
        self,
        mask_weight: float = 1.0,
        centroid_weight: float = 3.0,
        sdt_weight: float = 0.5,
        boundary_weight: float = 1.0,
        centroid_pos_weight: float = 10.0,
        centroid_min_target: float = 0.1,
        # Hybrid (SDT+boundary) stabilizers
        hybrid_use_tanh: bool = True,
        hybrid_bg_weight: float = 0.10,
        hybrid_interior_weight: float = 3.00,
        hybrid_boundary_weight: float = 1.00,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.centroid_weight = centroid_weight
        self.sdt_weight = sdt_weight
        self.boundary_weight = boundary_weight
        self.centroid_pos_weight = centroid_pos_weight
        self.centroid_min_target = centroid_min_target
        self.hybrid_use_tanh = hybrid_use_tanh
        self.hybrid_bg_weight = hybrid_bg_weight
        self.hybrid_interior_weight = hybrid_interior_weight
        self.hybrid_boundary_weight = hybrid_boundary_weight

    def forward(self, pred, target, buffer=None):
        if buffer is None:
            buffer = torch.ones_like(target[:, 0:1], dtype=torch.bool)

        buffer = buffer.squeeze(1)  # -> [B, h, w]

        losses = []

        if pred.shape[1] >= 1:  # mask
            mask_loss = self._mask_loss(pred[:, 0], target[:, 0], buffer)
            losses.append(self.mask_weight * mask_loss)

        if pred.shape[1] >= 2:  # centroid
            centroid_logits = pred[:, 1]
            centroid_target = target[:, 1]

            # use positively weighted MSE for centroid channel
            centroid_loss = weighted_centroid_mse_loss(
                centroid_logits,
                centroid_target,
                pos_weight=self.centroid_pos_weight,
                min_target=self.centroid_min_target,
            )
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
        """Hybrid SDT+boundary loss with region weighting.

        Target semantics:
          - background: 0
          - interior (inside crowns): (0, 1]
          - boundary: -1

        We downweight background (dominant), upweight interior (rare), and keep boundary moderate.
        Optionally apply tanh to constrain predictions to (-1, 1).
        """
        buffer = buffer.bool()

        # Optional stabilization: constrain regression output range
        if getattr(self, "hybrid_use_tanh", False):
            pred_eff = torch.tanh(pred)
        else:
            pred_eff = pred

        bg_mask = (target == 0) & buffer
        interior_mask = (target > 0) & buffer
        boundary_mask = (target == -1) & buffer

        # Weighted Smooth L1 for background + interior (both are non-boundary SDT regions)
        losses = []
        weights = []

        if bg_mask.any():
            bg_loss = F.smooth_l1_loss(pred_eff[bg_mask], target[bg_mask])
            losses.append(bg_loss)
            weights.append(float(getattr(self, "hybrid_bg_weight", 0.10)))

        if interior_mask.any():
            in_loss = F.smooth_l1_loss(pred_eff[interior_mask], target[interior_mask])
            losses.append(in_loss)
            weights.append(float(getattr(self, "hybrid_interior_weight", 3.00)))

        if len(losses) > 0:
            w = torch.tensor(weights, device=pred.device, dtype=torch.float32)
            sdt_loss = (torch.stack(losses) * w).sum() / (w.sum() + 1e-8)
        else:
            sdt_loss = torch.tensor(0.0, device=pred.device)

        # Boundary loss (L1) with its own internal weight
        if boundary_mask.any():
            bd_loss = F.l1_loss(pred_eff[boundary_mask], target[boundary_mask])
            boundary_loss = float(getattr(self, "hybrid_boundary_weight", 1.00)) * bd_loss
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


def weighted_centroid_mse_loss(logits, target, pos_weight: float = 10.0, min_target: float = 0.0):
    pred = torch.sigmoid(logits)

    w = torch.ones_like(target)

    if min_target > 0.0:
        mask_pos = target > min_target
    else:
        mask_pos = target > 0.0

    w[mask_pos] = pos_weight

    loss = w * (pred - target.float())**2
    return loss.sum() / w.sum()


def ewc_loss(model, fisher_information, optimal_parameters, lambda_ewc):
    ewc_reg = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            ewc_reg += (fisher_information[name] * (param - optimal_parameters[name]) ** 2).sum()
    return lambda_ewc * ewc_reg