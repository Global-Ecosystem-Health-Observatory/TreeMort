from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.metrics import log_metrics

logger = get_logger(__name__)


def trainer(
    model,
    optimizer,
    schedular,
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
            model, optimizer, schedular, criterion, metrics, 
            train_loader, conf, device
        )

        logger.info(f"[Train] Loss: {train_loss:.4f}")
        log_metrics(train_metrics, "Train")

        val_loss, val_metrics = validate_one_epoch(
            model, criterion, metrics, 
            val_loader, conf, device
        )

        logger.info(f"[Val] Loss: {val_loss:.4f}")
        log_metrics(val_metrics, "Val")

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