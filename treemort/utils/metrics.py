import torch


def iou_score(pred_probs, target, threshold=0.5):
    pred_binary = (pred_probs > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return torch.tensor(0.0)

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def f_score(pred_probs, target, threshold=0.5, beta=1):
    pred = (pred_probs > threshold).float()
    target = target.float()

    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()

    f_score = ((1 + beta**2) * tp + 1e-6) / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-6)
    return f_score