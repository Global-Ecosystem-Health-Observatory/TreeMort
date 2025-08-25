import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import weighted_dice_loss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation, proximity_metrics

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
            pred_channels = pred.shape[1]
            target_channels = target.shape[1]

            pred_mask = pred[:, 0, :, :]
            true_mask = target[:, 0, :, :]

            if target_channels > 3:
                buffer_mask = target[:, 3, :, :]
            else:
                buffer_mask = torch.ones_like(true_mask)

            pred_probs = apply_activation(pred_mask, activation=conf.activation)

            # Segmentation-level metrics
            iou_segments = masked_iou(pred_probs, true_mask, buffer_mask, threshold=conf.segment_threshold)
            f_score_segments = masked_f1(pred_probs, true_mask, buffer_mask, threshold=conf.segment_threshold)

            # Pixel-level metrics
            pred_bin = (pred_probs > conf.segment_threshold).float() * buffer_mask
            true_bin = (true_mask > conf.segment_threshold).float() * buffer_mask
            intersection = (pred_bin * true_bin).sum()
            pred_area = pred_bin.sum()
            true_area = true_bin.sum()
            union = ((pred_bin + true_bin) > 0).float().sum()
            pixel_precision = intersection / (pred_area + 1e-8)
            pixel_recall = intersection / (true_area + 1e-8)
            pixel_f1_score = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-8)

            if pred_channels > 1 and target_channels > 1:
                pred_centroid = pred[:, 1, :, :]
                true_centroid = target[:, 1, :, :]
                pred_centroid_probs = apply_activation(pred_centroid, activation=conf.activation)

                # Instance-level metrics (centroid-based)
                # Use proximity_metrics to get instance precision/recall/f1 and centroid error
                prox = proximity_metrics(
                    pred_centroid_probs,
                    true_centroid,
                    buffer_mask=buffer_mask,
                    proximity_threshold=5,
                    threshold=0.1,
                    min_distance=5
                )
                instance_precision = prox["precision"]
                instance_recall = prox["recall"]
                instance_f1_score = prox["f1_score"]
                centroid_err = prox["localization_error"]
            else:
                instance_precision = torch.tensor(0.0, device=pred_mask.device)
                instance_recall = torch.tensor(0.0, device=pred_mask.device)
                instance_f1_score = torch.tensor(0.0, device=pred_mask.device)
                centroid_err = torch.tensor(float('inf'), device=pred_mask.device)

            return {
                "iou_segments": iou_segments,
                "f_score_segments": f_score_segments,
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "pixel_f1_score": pixel_f1_score,
                "instance_precision": instance_precision,
                "instance_recall": instance_recall,
                "instance_f1_score": instance_f1_score,
                "centroid_err": centroid_err,
            }

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
                "mse_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.segment_threshold),
                "mse_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.segment_threshold),
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
                "iou_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.segment_threshold),
                "iou_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.segment_threshold),
            }
        logger.info("Buffer-weighted Dice loss configured.")
        return criterion, metrics

    else:
        raise ValueError(f"Unsupported loss type: {conf.loss}")