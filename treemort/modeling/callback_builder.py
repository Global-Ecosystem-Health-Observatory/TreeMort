import os

from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_callbacks(n_batches, output_dir, optimizer):
    checkpoint_dir = os.path.join(output_dir, "Checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "cp-{epoch:04d}.weights.pth"),
        save_freq=5 * n_batches,
        verbose=1,
    )

    best_checkpoint_cb = ModelCheckpoint(
        os.path.join(output_dir, "best.weights.pth"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr_cb = ReduceLROnPlateau(
        optimizer=optimizer,
        monitor="val_loss",
        mode="min",
        factor=0.7,
        patience=6,
        min_lr=8e-6,
        verbose=1,
    )

    early_stop_cb = EarlyStopping(
        patience=25,
        mode="min",
        verbose=1
    )

    return [checkpoint_cb, best_checkpoint_cb, reduce_lr_cb, early_stop_cb]