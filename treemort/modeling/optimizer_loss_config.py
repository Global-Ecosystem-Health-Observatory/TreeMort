import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from treemort.utils.loss import hybrid_loss, mse_loss, weighted_dice_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import iou_score, f_score, proximity_metrics, masked_iou, masked_f1, masked_mse

logger = get_logger(__name__)


def configure_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"{optimizer.__class__.__name__} optimizer configured with learning rate {learning_rate}.")
    return optimizer


def configure_loss_and_metrics(conf, class_weights=None):
    assert conf.loss in ["mse", "hybrid", "weighted_dice_loss"], "Invalid loss"

    if conf.loss == "hybrid":
        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)
            true_mask = target[:, 0, :, :].unsqueeze(1)
            true_centroid = target[:, 1, :, :].unsqueeze(1)
            true_hybrid = target[:, 2, :, :].unsqueeze(1)

            mask_loss = hybrid_loss(
                pred[:, 0].unsqueeze(1),  # Predicted mask
                true_mask,
                buffer_mask,
                class_weights=class_weights,
                dice_weight=0.5
            )

            centroid_loss = hybrid_loss(
                pred[:, 1].unsqueeze(1),  # Predicted centroids
                true_centroid,
                buffer_mask,
                dice_weight=0.0
            )

            sdt_loss = F.mse_loss(
                pred[:, 2].unsqueeze(1) * buffer_mask,
                true_hybrid * buffer_mask
            )

            return mask_loss + centroid_loss + 0.3 * sdt_loss

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]
            true_centroid = target[:, 1, :, :]

            seg_metrics = {
                "iou_segments": masked_iou(pred[:, 0], true_mask, buffer_mask),
                "f_score_segments": masked_f1(pred[:, 0], true_mask, buffer_mask)
            }
            
            centroid_metrics = proximity_metrics(
                pred[:, 1], true_centroid, buffer_mask,
                proximity_threshold=5, threshold=0.1
            )
            
            return {**seg_metrics, **centroid_metrics}

        logger.info("Configured hybrid loss with buffer masking")

    elif conf.loss == "mse":

        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)

            seg_loss = buffer_mask * mse_loss(pred[:, 0, :, :], target[:, 0, :, :])
            point_loss = buffer_mask * mse_loss(pred[:, 1, :, :], target[:, 1, :, :])

            valid_pixels = buffer_mask.sum() + 1e-8
            return (seg_loss.sum() + point_loss.sum()) / valid_pixels

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            return {
                "mse_segments": masked_mse(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask),
                "mse_points": masked_mse(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask),
            }

        logger.info("Masked MSE loss configured with buffer weighting")

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
                "iou_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask),
                "iou_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask),
            }

        logger.info("Buffer-weighted Dice loss configured")

    return criterion, metrics
