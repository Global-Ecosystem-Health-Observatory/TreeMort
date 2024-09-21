import logging

from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def handle_callbacks(callbacks, epoch, model, optimizer, val_loss):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            logger.info("Invoking ModelCheckpoint callback.")
            callback(epoch + 1, model, optimizer, val_loss)

        elif isinstance(callback, ReduceLROnPlateau):
            logger.info("Invoking ReduceLROnPlateau callback.")
            callback(val_loss)

        elif isinstance(callback, EarlyStopping):
            logger.info("Invoking EarlyStopping callback.")
            callback(epoch + 1, val_loss)
