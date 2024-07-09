import os
import math

import tensorflow as tf

from treeseg.utils.callbacks import build_callbacks


def trainer(model, train_dataset, val_dataset, num_train_samples, conf, output_dir):
    num_val_samples = int(conf.val_size * num_train_samples)
    num_train_batches = math.ceil(
        (num_train_samples - num_val_samples) / conf.train_batch_size
    )

    callbacks = build_callbacks(num_train_batches, output_dir)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=num_train_batches,
        epochs=conf.epochs,
        validation_steps=num_val_samples,
        callbacks=callbacks,
    )
