import random
import torch

import torchvision.transforms.functional as TF


def random_flip(image, label):
    if random.random() > 0.5:
        image = torch.flip(image, [2])  # Horizontal flip
        label = torch.flip(label, [2])
    if random.random() > 0.5:
        image = torch.flip(image, [1])  # Vertical flip
        label = torch.flip(label, [1])
    return image, label


def random_rotation(image, label):
    k = random.randint(0, 3)
    image = torch.rot90(image, k, [1, 2])
    label = torch.rot90(label, k, [1, 2])
    return image, label


def random_brightness(image, label):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    image = torch.clamp(image * factor, 0, 1)
    return image, label


def random_contrast(image, label):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    image = torch.clamp((image - mean) * factor + mean, 0, 1)
    return image, label


def random_multiplicative_noise(image, label):
    noise = torch.rand_like(image) * 0.2 + 0.9
    image = torch.clamp(image * noise, 0, 1)
    return image, label


def random_gamma(image, label):
    gamma = random.uniform(0.8, 1.2)
    image = torch.clamp(image**gamma, 0, 1)
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
