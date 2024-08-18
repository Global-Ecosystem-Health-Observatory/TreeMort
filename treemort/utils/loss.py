import torch
import torch.nn as nn


def dice_loss(pred, target):
    smooth = 1.0

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def hybrid_loss(pred, target):
    return dice_loss(pred, target) + focal_loss(pred, target)


def weighted_cross_entropy_dice_loss(pred, target, dice_weight=0.5):
    weights = torch.tensor([0.1, 0.9]).to(pred.device)  # Example weights, adjust as needed
    ce_loss = nn.CrossEntropyLoss(weight=weights)(pred, target)
    dice = dice_loss(pred, target)
    return dice_weight * dice + (1 - dice_weight) * ce_loss


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def f_score(pred, target, threshold=0.5, beta=1):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    f_score = ((1 + beta**2) * tp + 1e-6) / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-6)
    return f_score
