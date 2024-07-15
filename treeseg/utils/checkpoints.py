import os
import torch

def get_checkpoint(model_weights, output_dir):
    checkpoint = None

    if model_weights == "best":
        checkpoint = os.path.join(output_dir, "best.weights.pth")
        if not os.path.exists(checkpoint):
            checkpoint = None

    elif model_weights == "latest":
        checkpoints = [f for f in os.listdir(os.path.join(output_dir, "Checkpoints")) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(output_dir, "Checkpoints", f)))
            checkpoint = os.path.join(output_dir, "Checkpoints", latest_checkpoint)
        else:
            checkpoint = None

    return checkpoint


'''
import os

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
'''