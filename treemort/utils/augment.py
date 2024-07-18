import random

import numpy as np


def random_flip(image, label):
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy()  # Horizontal flip
        label = np.flip(label, axis=1).copy()
    if random.random() > 0.5:
        image = np.flip(image, axis=0).copy()  # Vertical flip
        label = np.flip(label, axis=0).copy()
    return image, label


def random_rotation(image, label):
    k = random.randint(0, 3)
    image = np.rot90(image, k).copy()
    label = np.rot90(label, k).copy()
    return image, label


def random_brightness(image, label):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    image = np.clip(image * factor, 0, 1).astype(
        np.float32
    )  # Adjust to normalized range
    return image, label


def random_contrast(image, label):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = np.clip((image - mean) * factor + mean, 0, 1).astype(
        np.float32
    )  # Adjust to normalized range
    return image, label


def random_multiplicative_noise(image, label):
    noise = np.random.uniform(0.9, 1.1, size=image.shape)
    image = np.clip(image * noise, 0, 1).astype(
        np.float32
    )  # Adjust to normalized range
    return image, label


def random_gamma(image, label):
    gamma = random.uniform(0.8, 1.2)
    image = np.clip((image**gamma), 0, 1).astype(
        np.float32
    )  # Adjust to normalized range
    return image, label


def apply_augmentations(image, label):
    image, label = random_flip(image, label)
    image, label = random_rotation(image, label)
    image, label = random_brightness(image, label)
    image, label = random_contrast(image, label)
    image, label = random_multiplicative_noise(image, label)
    image, label = random_gamma(image, label)
    return image, label


class Augmentations:
    def __call__(self, image, label):
        return apply_augmentations(image, label)
