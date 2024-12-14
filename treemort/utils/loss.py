import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits)
    
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    dice_score = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)
    dice_loss = 1 - dice_score
    return dice_loss


def weighted_dice_loss(logits, target, class_weights, smooth=1.0):
    probs = torch.sigmoid(logits)
    target = target.float()

    intersection = (probs * target).sum(dim=(2, 3))
    union = (probs + target).sum(dim=(2, 3))

    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score

    weights = target * class_weights[1] + (1 - target) * class_weights[0]
    weights = weights.mean(dim=(2, 3))

    weighted_dice_loss = (dice_loss * weights).mean()
    weighted_dice_loss = torch.clamp(weighted_dice_loss, min=0)
    return weighted_dice_loss


def focal_loss(logits, target, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
    
    probas = torch.sigmoid(logits)
    
    pt = torch.where(target == 1, probas, 1 - probas)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def hybrid_loss(logits, target, dice_weight=0.5, alpha=0.25, gamma=2, class_weights=None, use_dice=True):
    if class_weights is not None:
        weight = torch.ones_like(target) * class_weights[1]
        weight[target == 0] = class_weights[0]
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float(), weight=weight)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float())

    computed_dice_loss = 0.0

    if use_dice:
        if class_weights is not None:
            computed_dice_loss = weighted_dice_loss(logits, target, class_weights)
        else:
            computed_dice_loss = dice_loss(logits, target)

    computed_focal_loss = focal_loss(logits, target, alpha=alpha, gamma=gamma)

    total_loss = dice_weight * computed_dice_loss + (1 - dice_weight) * computed_focal_loss + bce_loss

    return total_loss


def mse_loss(logits, target):
    pred = torch.sigmoid(logits)
    return F.mse_loss(pred, target.float())
