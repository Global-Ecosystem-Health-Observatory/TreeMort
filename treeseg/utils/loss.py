import torch
import torch.nn as nn


def dice_loss(pred, target):
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def focal_loss(pred, target, alpha=0.8, gamma=2):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss


def hybrid_loss(pred, target):
    return dice_loss(pred, target) + focal_loss(pred, target)


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
    f_score = ((1 + beta**2) * tp + 1e-6) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + 1e-6
    )
    return f_score
