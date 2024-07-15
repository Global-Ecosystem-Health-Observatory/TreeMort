import os

import tensorflow as tf


def build_callbacks(n_batches, output_dir):
    checkpoint_dir = os.path.join(output_dir, "Checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "cp-{epoch:04d}.weights.h5"),
        save_weights_only=True,
        save_freq=5 * n_batches,
        verbose=1,
    )

    best_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best.weights.h5"),
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.7, patience=6, min_lr=8e-6, verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        patience=25, restore_best_weights=False, verbose=1
    )

    callbacks = [
        checkpoint_cb,
        best_checkpoint_cb,
        reduce_lr_cb,
        early_stop_cb,
    ]

    return callbacks
