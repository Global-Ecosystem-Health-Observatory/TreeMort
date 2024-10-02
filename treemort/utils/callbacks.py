import torch

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCheckpoint:
    def __init__(
        self,
        filepath,
        save_weights_only=True,
        save_freq=1,
        monitor=None,
        mode="min",
        save_best_only=False,
        verbose=1,
    ):
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = None
        if self.mode == "min":
            self.best = float("inf")
        elif self.mode == "max":
            self.best = -float("inf")

    def __call__(self, epoch, model, optimizer, val_loss=None):
        if self.save_best_only:
            if (self.mode == "min" and val_loss < self.best) or (
                self.mode == "max" and val_loss > self.best
            ):
                self.best = val_loss
                if self.verbose:
                    logger.info(f"Saving best model with {self.monitor}: {val_loss}")
                torch.save(model.state_dict(), self.filepath)
        else:
            if epoch % self.save_freq == 0:
                if self.verbose:
                    logger.info(f"Saving model at epoch {epoch}")
                torch.save(model.state_dict(), self.filepath.format(epoch=epoch))


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    ):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = param_group["lr"] * self.factor
                if new_lr >= self.min_lr:
                    param_group["lr"] = new_lr
                    if self.verbose:
                        logger.info(f"Reducing learning rate to {new_lr}")
                else:
                    param_group["lr"] = self.min_lr
            self.num_bad_epochs = 0


class EarlyStopping:
    def __init__(self, patience=10, verbose=1):
        self.patience = patience
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def __call__(self, epoch, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            self.stop_training = True
            if self.verbose:
                logger.info(f"Early stopping at epoch {epoch}")
