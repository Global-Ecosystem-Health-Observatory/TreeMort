import random
from typing import Tuple

import torch
import torch.nn.functional as F
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


def random_scale_jitter(image, label, scale_range: Tuple[float, float], prob: float):
    min_scale, max_scale = scale_range
    if prob <= 0.0 or max_scale <= 0.0 or (abs(min_scale - 1.0) < 1e-6 and abs(max_scale - 1.0) < 1e-6):
        return image, label
    if random.random() > prob:
        return image, label
    h, w = image.shape[1:]
    scale = random.uniform(min_scale, max_scale)
    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))
    image_down = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    label_down = F.interpolate(label.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    image = F.interpolate(image_down.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    label = F.interpolate(label_down.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return image, label


def random_brightness(image, label, max_jitter: float):
    if max_jitter <= 0.0:
        return image, label
    factor = 1.0 + random.uniform(-max_jitter, max_jitter)
    image = torch.clamp(image * factor, 0, 1)
    return image, label


def random_contrast(image, label, max_jitter: float):
    if max_jitter <= 0.0:
        return image, label
    factor = 1.0 + random.uniform(-max_jitter, max_jitter)
    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    image = torch.clamp((image - mean) * factor + mean, 0, 1)
    return image, label


def random_hue_saturation(image, label, hue_jitter: float, saturation_jitter: float):
    if image.shape[0] < 3 or (hue_jitter <= 0 and saturation_jitter <= 0):
        return image, label
    hue_factor = random.uniform(-hue_jitter, hue_jitter) if hue_jitter > 0 else 0.0
    sat_factor = 1.0 + random.uniform(-saturation_jitter, saturation_jitter) if saturation_jitter > 0 else 1.0
    rgb = image[:3]
    rgb = TF.adjust_hue(rgb, hue_factor)
    rgb = TF.adjust_saturation(rgb, sat_factor)
    image = torch.cat([rgb, image[3:]], dim=0) if image.shape[0] > 3 else rgb
    image = torch.clamp(image, 0, 1)
    return image, label


def random_multiplicative_noise(image, label, noise_range: Tuple[float, float]):
    low, high = noise_range
    if high <= 0 or abs(high - 1.0) < 1e-6 and abs(low - 1.0) < 1e-6:
        return image, label
    noise = torch.empty_like(image).uniform_(low, high)
    image = torch.clamp(image * noise, 0, 1)
    return image, label


def random_gamma(image, label, gamma_range: Tuple[float, float]):
    gamma_min, gamma_max = gamma_range
    if gamma_max <= 0 or abs(gamma_min - 1.0) < 1e-6 and abs(gamma_max - 1.0) < 1e-6:
        return image, label
    gamma = random.uniform(gamma_min, gamma_max)
    image = torch.clamp(image**gamma, 0, 1)
    return image, label


def random_gaussian_blur(image, label, prob: float):
    if prob <= 0.0 or random.random() > prob:
        return image, label
    kernel_size = random.choice([3, 5])
    sigma = random.uniform(0.1, 1.2)
    image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
    return image, label


def apply_augmentations(image, label, params):
    image, label = random_flip(image, label)
    image, label = random_rotation(image, label)
    image, label = random_scale_jitter(image, label, params["scale_range"], params["scale_prob"])
    image, label = random_brightness(image, label, params["brightness"])
    image, label = random_contrast(image, label, params["contrast"])
    image, label = random_hue_saturation(image, label, params["hue"], params["saturation"])
    image, label = random_multiplicative_noise(image, label, params["noise_range"])
    image, label = random_gamma(image, label, params["gamma_range"])
    image, label = random_gaussian_blur(image, label, params["blur_prob"])
    return image, label


class Augmentations:
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        gamma_range: Tuple[float, float] = (1.0, 1.0),
        hue: float = 0.0,
        saturation: float = 0.0,
        noise_range: Tuple[float, float] = (1.0, 1.0),
        scale_range: Tuple[float, float] = (1.0, 1.0),
        scale_prob: float = 0.0,
        blur_prob: float = 0.0,
    ):
        self.params = {
            "brightness": max(brightness, 0.0),
            "contrast": max(contrast, 0.0),
            "gamma_range": gamma_range,
            "hue": max(hue, 0.0),
            "saturation": max(saturation, 0.0),
            "noise_range": noise_range,
            "scale_range": scale_range,
            "scale_prob": max(scale_prob, 0.0),
            "blur_prob": max(blur_prob, 0.0),
        }

    def __call__(self, image, label):
        return apply_augmentations(image, label, self.params)
