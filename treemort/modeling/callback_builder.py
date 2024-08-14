import os

from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_callbacks(n_batches, output_dir, optimizer):
    checkpoint_dir = os.path.join(output_dir, "Checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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
        factor=0.7,
        patience=6,
        min_lr=8e-6,
        verbose=1,
    )

    early_stop_cb = EarlyStopping(patience=25, verbose=1)

    callbacks = [
        checkpoint_cb,
        best_checkpoint_cb,
        reduce_lr_cb,
        early_stop_cb,
    ]

    return callbacks
