from treemort.utils.logger import get_logger
from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

logger = get_logger(__name__)


def handle_callbacks(callbacks, epoch, model, optimizer, val_loss, val_metrics=None):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            logger.info("Invoking ModelCheckpoint callback")
            current_value = val_loss
            if val_metrics and callback.monitor in val_metrics:
                current_value = val_metrics[callback.monitor]
            
            callback(epoch + 1, model, optimizer, val_loss=current_value)

        elif isinstance(callback, ReduceLROnPlateau):
            logger.info("Invoking ReduceLROnPlateau callback")
            callback(val_loss)

        elif isinstance(callback, EarlyStopping):
            logger.info("Invoking EarlyStopping callback")
            callback(epoch + 1, val_loss)

        else:
            logger.warning(f"Unknown callback type: {type(callback)}")