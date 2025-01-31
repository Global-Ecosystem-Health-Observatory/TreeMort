import torch

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCheckpoint:
    def __init__(self, filepath, save_weights_only=True, save_freq=1, 
                 monitor='val_loss', mode="min", save_best_only=False, verbose=1):
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
        self.monitor = monitor

    def __call__(self, epoch, model, optimizer, val_loss=None):
        current_value = val_loss
        
        if self.monitor != 'val_loss' and hasattr(self, 'val_metrics'):
            current_value = self.val_metrics.get(self.monitor, val_loss)

        if self.save_best_only:
            if (self.mode == "min" and current_value < self.best) or \
               (self.mode == "max" and current_value > self.best):
                self.best = current_value
                if self.verbose:
                    logger.info(f"Saving best model with {self.monitor}: {current_value}")
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
        mode="min",
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    ):
        self.optimizer = optimizer
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0

    def __call__(self, current_value):
        if self.best is None:
            self.best = current_value
            self.num_bad_epochs = 0
            return

        if self.mode == "min":
            is_better = current_value < self.best
        else:  # mode == "max"
            is_better = current_value > self.best

        if is_better:
            self.best = current_value
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
    def __init__(self, patience=10, mode="min", verbose=1):  # Add mode
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0
        self.stop_training = False

    def __call__(self, epoch, current_value):
        if self.best is None:
            self.best = current_value
            return

        if self.mode == "min":
            is_better = current_value < self.best
        else:
            is_better = current_value > self.best

        if is_better:
            self.best = current_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.stop_training = True
            if self.verbose:
                logger.info(f"Early stopping at epoch {epoch}")