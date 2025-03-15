import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import weighted_dice_loss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation

logger = get_logger(__name__)


def configure_optimizer(model, learning_rate, total_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3
    )
    
    logger.info(f"Configured {optimizer.__class__.__name__} with OneCycleLR scheduler (max_lr={learning_rate}, total_steps={total_steps}).")
    return optimizer, scheduler


def configure_loss_and_metrics(conf, class_weights=None):
    if conf.loss == "hybrid":
        def criterion(pred, target):
            pred_mask = pred[:, 0, :, :]

            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]

            return hybrid_loss(pred_mask, true_mask, buffer_mask)

        def metrics(pred, target):
            pred_mask = pred[:, 0, :, :]

            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]

            pred_probs = apply_activation(pred_mask, activation=conf.activation)

            seg_metrics = {
                "iou_segments": masked_iou(pred_probs, true_mask, buffer_mask, threshold=conf.threshold),
                "f_score_segments": masked_f1(pred_probs, true_mask, buffer_mask, threshold=conf.threshold)
            }

            return {**seg_metrics}

        logger.info("Configured hybrid loss using TreeMortalityLoss class (BCE, MSE, and L1-based hybrid loss).")
        return criterion, metrics

    elif conf.loss == "mse":
        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)
            seg_loss = buffer_mask * F.mse_loss(pred[:, 0, :, :], target[:, 0, :, :])
            point_loss = buffer_mask * F.mse_loss(pred[:, 1, :, :], target[:, 1, :, :])
            valid_pixels = buffer_mask.sum() + 1e-8
            return (seg_loss.sum() + point_loss.sum()) / valid_pixels

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            return {
                "mse_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.threshold),
                "mse_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.threshold),
            }
        logger.info("Masked MSE loss configured with buffer weighting.")
        return criterion, metrics

    elif conf.loss == "weighted_dice_loss":
        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)
            seg_loss = weighted_dice_loss(
                pred[:, 0, :, :], target[:, 0, :, :], buffer_mask=buffer_mask, class_weights=class_weights
            )
            centroid_loss = weighted_dice_loss(
                pred[:, 1, :, :], target[:, 1, :, :], buffer_mask=buffer_mask, class_weights=class_weights
            )
            return seg_loss + centroid_loss

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            return {
                "iou_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.threshold),
                "iou_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.threshold),
            }
        logger.info("Buffer-weighted Dice loss configured.")
        return criterion, metrics

    else:
        raise ValueError(f"Unsupported loss type: {conf.loss}")