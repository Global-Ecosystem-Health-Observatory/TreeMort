import os
import platform

import tensorflow as tf


def get_checkpoint(model_weights, output_dir):
    checkpoint_dir = os.path.join(output_dir, "Checkpoints")

    checkpoint = None
    if model_weights == "best":
        checkpoint = os.path.join(checkpoint_dir, "best.weights.h5")

    elif model_weights == "latest":
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    return checkpoint
