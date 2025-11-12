from treemort.utils.logger import get_logger
from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, EncoderFreezeCallback
from treemort.utils.wandb_utils import log_model_artifact, format_aliases

logger = get_logger(__name__)


def _float_value(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _primary_metric(val_metrics, fallback_loss):
    keys = ["tree_iou", "iou_segments", "pixel_f1_score", "instance_f1_score"]
    for key in keys:
        if val_metrics and key in val_metrics and val_metrics[key] is not None:
            return _float_value(val_metrics[key])
    if fallback_loss is None:
        return None
    return 1.0 / (1.0 + fallback_loss)


def handle_callbacks(
    callbacks,
    epoch,
    model,
    optimizer,
    val_loss,
    val_metrics=None,
    wandb_run=None,
    conf=None,
    dataset_meta=None,
):
    dataset_meta = dataset_meta or {}
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            logger.info("Invoking ModelCheckpoint callback")
            current_value = val_loss
            if val_metrics and callback.monitor in val_metrics:
                current_value = val_metrics[callback.monitor]
            
            callback(epoch + 1, model, optimizer, val_loss=current_value)

            if (
                getattr(callback, "just_saved", False)
                and wandb_run is not None
                and conf is not None
                and getattr(conf, "wandb", False)
            ):
                artifact_name = f"model-{conf.model}"
                aliases = format_aliases(getattr(wandb_run, "id", ""), callback.alias)
                metric_value = _primary_metric(val_metrics, val_loss)
                promote_threshold = getattr(conf, "promote_threshold", 0.0) or 0.0
                if (
                    callback.alias == "best"
                    and promote_threshold > 0
                    and metric_value is not None
                    and metric_value >= promote_threshold
                ):
                    aliases.append("candidate")

                metadata = {
                    "model": conf.model,
                    "checkpoint_alias": callback.alias,
                    "epoch": epoch,
                    "val_loss": _float_value(val_loss),
                    "primary_metric": metric_value,
                    "dataset_artifact": getattr(conf, "dataset_artifact", None),
                    "dataset_meta": {k: str(v) for k, v in dataset_meta.items()},
                }
                if val_metrics:
                    metadata["val_metrics"] = {
                        key: _float_value(value)
                        for key, value in val_metrics.items()
                        if value is not None
                    }

                unique_aliases = []
                for alias in aliases:
                    if alias and alias not in unique_aliases:
                        unique_aliases.append(alias)

                log_model_artifact(
                    wandb_run,
                    callback.last_save_path,
                    artifact_name,
                    unique_aliases,
                    metadata=metadata,
                )

        elif isinstance(callback, ReduceLROnPlateau):
            logger.info("Invoking ReduceLROnPlateau callback")
            callback(val_loss)

        elif isinstance(callback, EarlyStopping):
            logger.info("Invoking EarlyStopping callback")
            callback(epoch + 1, val_loss)

        elif isinstance(callback, EncoderFreezeCallback):
            # Encoder freeze callback is called at epoch start within trainer; nothing to do here.
            continue

        else:
            logger.warning(f"Unknown callback type: {type(callback)}")
