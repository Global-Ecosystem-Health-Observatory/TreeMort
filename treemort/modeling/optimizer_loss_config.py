import torch
import torch.nn as nn
import torch.optim as optim

from treemort.utils.loss import hybrid_loss, mse_loss, weighted_dice_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import iou_score, f_score

logger = get_logger(__name__)


def configure_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"{optimizer.__class__.__name__} optimizer configured with learning rate {learning_rate}.")
    return optimizer


def configure_loss_and_metrics(conf):
    assert conf.loss in [
        "mse",
        "hybrid",
        "weighted_dice_loss",
    ], f"[ERROR] Invalid loss function: {conf.loss}."

    if conf.loss == "hybrid": # TODO. plateau-ed in local minima
        criterion = hybrid_loss
        metrics = lambda pred, target: {
            "iou_score": iou_score(pred, target, conf.threshold),
            "f_score": f_score(pred, target, conf.threshold),
        }
        logger.info("Hybrid loss and metrics (IOU, F-Score) configured.")
    elif conf.loss == "mse":
        criterion = mse_loss
        metrics = lambda pred, target: {
            "mse": mse_loss(pred, target),
            "mae": nn.functional.l1_loss(pred, target),
            "rmse": torch.sqrt(mse_loss(pred, target)),
        }
        logger.info("MSE loss and metrics (MSE, MAE, RMSE) configured.")
    elif conf.loss == "weighted_dice_loss":
        criterion = weighted_dice_loss
        metrics = lambda pred, target: {
            "iou_score": iou_score(pred, target, conf.threshold),
            "f_score": f_score(pred, target, conf.threshold),
        }
        logger.info("Weighted Dice loss and metrics (IOU, F-Score) configured.")

    return criterion, metrics
