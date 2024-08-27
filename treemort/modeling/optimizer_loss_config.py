import torch
import torch.nn as nn
import torch.optim as optim

from treemort.utils.loss import hybrid_loss, mse_loss, iou_score, f_score, weighted_dice_loss


def configure_optimizer(model, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(
        f"[INFO] {optimizer.__class__.__name__} optimizer configured with learning rate {learning_rate}."
    )
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
    elif conf.loss == "mse":
        criterion = mse_loss
        metrics = lambda pred, target: {
            "mse": mse_loss(pred, target),
            "mae": nn.functional.l1_loss(pred, target),
            "rmse": torch.sqrt(mse_loss(pred, target)),
        }
    elif conf.loss == "weighted_dice_loss":
        criterion = weighted_dice_loss
        metrics = lambda pred, target: {
            "iou_score": iou_score(pred, target, conf.threshold),
            "f_score": f_score(pred, target, conf.threshold),
        }

    print(f"[INFO] Loss function '{conf.loss}' and associated metrics configured.")
    return criterion, metrics
