from __future__ import annotations

import torch

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCheckpoint:
    def __init__(
        self,
        filepath,
        save_weights_only=True,
        save_freq=1,
        monitor='val_loss',
        mode="min",
        save_best_only=False,
        verbose=1,
        alias: str | None = None,
    ):
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.alias = alias
        self.best = None
        if self.mode == "min":
            self.best = float("inf")
        elif self.mode == "max":
            self.best = -float("inf")
        self.monitor = monitor
        self.just_saved = False
        self.last_save_path = None

    def __call__(self, epoch, model, optimizer, val_loss=None):
        current_value = val_loss
        
        if self.monitor != 'val_loss' and hasattr(self, 'val_metrics'):
            current_value = self.val_metrics.get(self.monitor, val_loss)

        self.just_saved = False

        if self.save_best_only:
            if (self.mode == "min" and current_value < self.best) or \
               (self.mode == "max" and current_value > self.best):
                self.best = current_value
                if self.verbose:
                    logger.info(f"Saving best model with {self.monitor}: {current_value}")
                torch.save(model.state_dict(), self.filepath)
                self.last_save_path = self.filepath
                self.just_saved = True
        else:
            if epoch % self.save_freq == 0:
                if self.verbose:
                    logger.info(f"Saving model at epoch {epoch}")
                save_path = self.filepath.format(epoch=epoch)
                torch.save(model.state_dict(), save_path)
                self.last_save_path = save_path
                self.just_saved = True


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


class EncoderFreezeCallback:
    def __init__(self, model, freeze_epochs: int, keep_blocks: int):
        self.model = model
        self.freeze_epochs = max(0, freeze_epochs)
        self.keep_blocks = max(0, keep_blocks)
        self._frozen = False
        self._unfrozen = False
        if self.enabled:
            self._toggle_blocks(False)
            self._frozen = True

    @property
    def enabled(self) -> bool:
        return self.freeze_epochs > 0 and self.keep_blocks > 0 and self._encoder_blocks

    @property
    def _encoder_blocks(self):
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return []
        blocks = list(encoder.children())
        return blocks[: min(len(blocks), self.keep_blocks)]

    def _toggle_blocks(self, requires_grad: bool):
        blocks = self._encoder_blocks
        if not blocks:
            logger.warning("EncoderFreezeCallback enabled but no encoder blocks found.")
            return
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = requires_grad
        state = "unfrozen" if requires_grad else "frozen"
        logger.info(f"EncoderFreezeCallback {state} first {len(blocks)} blocks.")

    def on_epoch_start(self, epoch: int):
        if not self.enabled or self._unfrozen:
            return
        if epoch >= self.freeze_epochs:
            self._toggle_blocks(True)
            self._unfrozen = True
