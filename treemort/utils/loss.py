import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


def weighted_dice_loss(pred, target, weights, smooth=1.0):
    target = target.squeeze(1).long()

    if target.max() >= pred.shape[1]:
        target = torch.clamp(target, max=pred.shape[1] - 1)

    target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3))

    dice_score = 2 * (intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score

    weighted_dice_loss = (dice_loss * weights).mean()
    return weighted_dice_loss


def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)  # pt is the probability of being correct
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def hybrid_loss(pred, target, dice_weight=0.5, alpha=0.9, gamma=2, class_weights=None):
    if class_weights is not None:
        weight = torch.ones_like(target) * class_weights[1]
        weight[target == 0] = class_weights[0]
        bce_loss = F.binary_cross_entropy(pred, target, weight=weight)

    else:
        bce_loss = F.binary_cross_entropy(pred, target)

    if class_weights is not None:
        dice = weighted_dice_loss(pred, target, class_weights)
    else:
        dice = dice_loss(pred, target)

    fl = focal_loss(pred, target, alpha=alpha, gamma=gamma)
    return dice_weight * dice + (1 - dice_weight) * fl + bce_loss


'''

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


def weighted_cross_entropy_dice_loss(pred, target, class_weights, dice_weight=0.5):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(pred, target)
    dice = dice_loss(pred, target)
    return dice_weight * dice + (1 - dice_weight) * ce_loss


def weighted_cross_entropy_dice_loss(pred, target, dice_weight=0.5):
    weights = torch.tensor([0.1, 0.9]).to(pred.device)  # Example weights, adjust as needed
    ce_loss = nn.CrossEntropyLoss(weight=weights)(pred, target)
    dice = dice_loss(pred, target)
    return dice_weight * dice + (1 - dice_weight) * ce_loss

'''
def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)
'''

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
'''

def iou_score(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 0.0

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def f_score(pred, target, threshold=0.5, beta=1):
    pred = (pred > threshold).float()

    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()

    f_score = ((1 + beta**2) * tp + 1e-6) / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-6)
    return f_score
