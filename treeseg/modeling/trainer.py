import os
import math

import tensorflow as tf

from treeseg.modeling.builder import build_model
from treeseg.utils.callbacks import build_callbacks
from treeseg.utils.checkpoints import get_checkpoint


def resume_or_load(conf):
    model = build_model(conf.model, conf.input_channels, conf.output_channels)

    checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

    if checkpoint:
        model.load_weights(checkpoint, skip_mismatch=True)
        print(f"Loaded weights from {checkpoint}")

    else:
        print("No checkpoint found. Proceeding without loading weights.")

        # assert not conf.eval_only, "The number of image and label paths should be the same."

    return model


def trainer(model, train_dataset, val_dataset, num_train_samples, conf):
    num_val_samples = int(conf.val_size * num_train_samples)
    num_train_batches = math.ceil(
        (num_train_samples - num_val_samples) / conf.train_batch_size
    )

    callbacks = build_callbacks(num_train_batches, conf.output_dir)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=num_train_batches,
        epochs=conf.epochs,
        validation_steps=num_val_samples,
        callbacks=callbacks,
    )
