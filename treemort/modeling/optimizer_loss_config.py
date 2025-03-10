import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import weighted_dice_loss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1

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
    if conf.loss == "mse":
        def criterion(pred, target):
            pred_mask = pred[:, 0, :, :]

            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]

            valid_pixels = buffer_mask.sum() + 1e-8

            seg_loss = (buffer_mask * F.mse_loss(pred_mask, true_mask))

            return seg_loss.sum() / valid_pixels

        def metrics(pred, target):
            pred_mask = pred[:, 0, :, :]

            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]

            return {
                "iou_segments": masked_iou(pred_mask, true_mask, buffer_mask, threshold=conf.threshold),
                "f_score_segments": masked_f1(pred_mask, true_mask, buffer_mask, threshold=conf.threshold),
            }
        logger.info("Masked MSE loss configured with buffer weighting.")
        return criterion, metrics

    else:
        raise ValueError(f"Unsupported loss type: {conf.loss}")