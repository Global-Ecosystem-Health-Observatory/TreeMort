import torch
import torch.nn as nn
import torch.optim as optim

from treemort.utils.loss import hybrid_loss, mse_loss, weighted_dice_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import iou_score, f_score, proximity_metrics

logger = get_logger(__name__)


def configure_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"{optimizer.__class__.__name__} optimizer configured with learning rate {learning_rate}.")
    return optimizer


def configure_loss_and_metrics(conf, class_weights=None):
    assert conf.loss in [
        "mse",
        "hybrid",
        "weighted_dice_loss",
    ], f"[ERROR] Invalid loss function: {conf.loss}."

    if conf.loss == "hybrid":
        def criterion(pred, target):
            seg_loss = hybrid_loss(
                pred[:, 0, :, :], 
                target[:, 0, :, :], 
                dice_weight=0.5, 
                class_weights=class_weights
            )

            point_loss = hybrid_loss(
                pred[:, 1, :, :], 
                target[:, 1, :, :], 
                dice_weight=0.0,
                class_weights=None,
                use_dice=False
            )

            return seg_loss + point_loss

        def metrics(pred, target):
            seg_metrics = {
                "iou_segments": iou_score(pred[:, 0, :, :], target[:, 0, :, :], conf.threshold),
                "f_score_segments": f_score(pred[:, 0, :, :], target[:, 0, :, :], conf.threshold),
            }
            
            centroid_metrics = proximity_metrics(
                pred[:, 1, :, :],
                target[:, 1, :, :],
                proximity_threshold=5,
                threshold=0.1,
                min_distance=5
            )
            
            return {**seg_metrics, **centroid_metrics}

        logger.info("Hybrid loss and metrics for segmentation and Gaussian centroid maps configured.")

    elif conf.loss == "mse":
        def criterion(pred, target):
            seg_loss = mse_loss(pred[:, 0, :, :], target[:, 0, :, :])
            point_loss = mse_loss(pred[:, 1, :, :], target[:, 1, :, :])
            return seg_loss + point_loss

        def metrics(pred, target):
            return {
                "mse_segments": mse_loss(pred[:, 0, :, :], target[:, 0, :, :]),
                "mae_segments": nn.functional.l1_loss(pred[:, 0, :, :], target[:, 0, :, :]),
                "rmse_segments": torch.sqrt(mse_loss(pred[:, 0, :, :])),
                "mse_points": mse_loss(pred[:, 1, :, :], target[:, 1, :, :]),
                "mae_points": nn.functional.l1_loss(pred[:, 1, :, :], target[:, 1, :, :]),
                "rmse_points": torch.sqrt(mse_loss(pred[:, 1, :, :])),
            }

        logger.info("MSE loss and metrics for multi-channel outputs configured.")

    elif conf.loss == "weighted_dice_loss":
        def criterion(pred, target):
            seg_loss = weighted_dice_loss(pred[:, 0, :, :], target[:, 0, :, :])
            point_loss = weighted_dice_loss(pred[:, 1, :, :], target[:, 1, :, :])
            return seg_loss + point_loss

        def metrics(pred, target):
            return {
                "iou_segments": iou_score(pred[:, 0, :, :], target[:, 0, :, :], conf.threshold),
                "f_score_segments": f_score(pred[:, 0, :, :], target[:, 0, :, :], conf.threshold),
                "iou_points": iou_score(pred[:, 1, :, :], target[:, 1, :, :], conf.threshold),
                "f_score_points": f_score(pred[:, 1, :, :], target[:, 1, :, :], conf.threshold),
            }

        logger.info("Weighted Dice loss and metrics for multi-channel outputs configured.")

    return criterion, metrics
