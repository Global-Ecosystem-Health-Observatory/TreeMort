from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.callbacks import EarlyStopping

logger = get_logger(__name__)


def trainer(
    model,
    optimizer,
    criterion,
    metrics,
    train_loader,
    val_loader,
    conf,
    callbacks,
):
    device = next(model.parameters()).device
    best_metric = float('inf')

    for epoch in range(conf.epochs):
        logger.info(f"\nEpoch {epoch+1}/{conf.epochs}")
        logger.info("-------------------------------")
        
        train_loss, train_metrics = train_one_epoch(
            model, optimizer, criterion, metrics, 
            train_loader, conf, device
        )

        logger.info(f"[Train] Loss: {train_loss:.4f}")
        _log_metrics(train_metrics, "Train")

        val_loss, val_metrics = validate_one_epoch(
            model, criterion, metrics, 
            val_loader, conf, device
        )

        logger.info(f"[Val] Loss: {val_loss:.4f}")
        _log_metrics(val_metrics, "Val")

        stop_training = handle_callbacks(
            callbacks,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_metrics=val_metrics
        )

        if stop_training:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training completed successfully.")
    return model


def _log_metrics(metrics, phase):
    seg_metrics = {k:v for k,v in metrics.items() if "segments" in k}
    logger.info(f"{phase} Segmentation Metrics:")
    for metric, value in seg_metrics.items():
        logger.info(f"  {metric.replace('_segments', '').title()}: {value:.4f}")

    cent_metrics = {k:v for k,v in metrics.items() if "points" in k}
    logger.info(f"{phase} Centroid Metrics:")
    for metric, value in cent_metrics.items():
        logger.info(f"  {metric.replace('_points', '').title()}: {value:.4f}")

    inst_metrics = {k:v for k,v in metrics.items() if "instance" in k}
    if inst_metrics:
        logger.info(f"{phase} Instance Metrics:")
        for metric, value in inst_metrics.items():
            logger.info(f"  {metric.replace('_instance', '').title()}: {value:.4f}")

    prox_metrics = {k:v for k,v in metrics.items() if "proximity" in k}
    if prox_metrics:
        logger.info(f"{phase} Proximity Metrics:")
        for metric, value in prox_metrics.items():
            logger.info(f"  {metric.title()}: {value:.4f}")