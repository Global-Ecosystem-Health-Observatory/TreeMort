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


def hybrid_loss(logits, target, dice_weight=0.5, alpha=0.25, gamma=2, class_weights=None):
    if class_weights is not None:
        weight = torch.ones_like(target) * class_weights[1]
        weight[target == 0] = class_weights[0]
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float(), weight=weight)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.float())

    if class_weights is not None:
        dice = weighted_dice_loss(logits, target, class_weights)
    else:
        dice = dice_loss(logits, target)

    fl = focal_loss(logits, target, alpha=alpha, gamma=gamma)

    total_loss = dice_weight * dice + (1 - dice_weight) * fl + bce_loss
    return total_loss


def mse_loss(logits, target):
    pred = torch.sigmoid(logits)
    return F.mse_loss(pred, target.float())


def ewc_loss(model, fisher_information, optimal_parameters, lambda_ewc):
    ewc_reg = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            ewc_reg += (fisher_information[name] * (param - optimal_parameters[name]) ** 2).sum()
    return lambda_ewc * ewc_reg
