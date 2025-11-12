from tqdm import tqdm

from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.metrics import log_metrics


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
    wandb_run=None,
    dataset_meta=None,
):
    logger = get_logger()

    device = next(model.parameters()).device
    best_metric = float('inf')
    dataset_meta = dataset_meta or {}

    def _prefixed_metrics(prefix, data):
        logged = {}
        if not data:
            return logged
        for key, value in data.items():
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            logged[f"{prefix}/{key}"] = float(value)
        return logged

    for epoch in tqdm(range(conf.epochs), desc="Epochs", unit="epoch"):

        for callback in callbacks:
            if hasattr(callback, "on_epoch_start"):
                callback.on_epoch_start(epoch)
        
        train_loss, train_metrics = train_one_epoch(
            model, optimizer, schedular, criterion, metrics, 
            train_loader, conf, device
        )

        # logger.info(f"[Train] Loss: {train_loss:.4f}")
        # log_metrics(train_metrics, "Train")

        val_loss, val_metrics = validate_one_epoch(
            model, criterion, metrics, 
            val_loader, conf, device
        )

        # logger.info(f"[Val] Loss: {val_loss:.4f}")
        # log_metrics(val_metrics, "Val")

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "dataset_artifact": getattr(conf, "dataset_artifact", None),
                "promote_threshold": getattr(conf, "promote_threshold", None),
            }
            payload.update(_prefixed_metrics("train", train_metrics))
            payload.update(_prefixed_metrics("val", val_metrics))
            try:
                wandb_run.log(payload)
            except Exception as exc:
                logger.warning(f"Failed to log metrics to W&B at epoch {epoch}: {exc}")

        stop_training = handle_callbacks(
            callbacks,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_metrics=val_metrics,
            wandb_run=wandb_run,
            conf=conf,
            dataset_meta=dataset_meta,
        )

        if stop_training:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training completed successfully.")
    return model
