import os
import platform

import tensorflow as tf


def get_checkpoint(model_weights, output_dir):
    checkpoint = None
    
    if model_weights == "best":
        checkpoint = os.path.join(output_dir, "best.weights.h5")

        if not os.path.exists(checkpoint):
            checkpoint = None

    elif model_weights == "latest":
        checkpoint = tf.train.latest_checkpoint(os.path.join(output_dir, "Checkpoints"))

    return checkpoint
