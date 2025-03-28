import pytest
import torch

from treemort.utils.augment import (
    random_flip,
    random_rotation,
    random_brightness,
    random_contrast,
    random_multiplicative_noise,
    random_gamma,
    apply_augmentations,
    Augmentations,
)

@pytest.fixture
def sample_image_label():
    image = torch.rand(3, 256, 256)  # Random image in (C, H, W) format
    label = torch.randint(0, 2, (1, 256, 256)).float()  # Random binary label in (C, H, W) format
    return image, label


def test_random_flip(sample_image_label):
    image, label = sample_image_label
    flipped_image, flipped_label = random_flip(image, label)
    assert flipped_image.shape == image.shape
    assert flipped_label.shape == label.shape


def test_random_rotation(sample_image_label):
    image, label = sample_image_label
    rotated_image, rotated_label = random_rotation(image, label)
    assert rotated_image.shape == image.shape
    assert rotated_label.shape == label.shape


def test_random_brightness(sample_image_label):
    image, label = sample_image_label
    bright_image, bright_label = random_brightness(image, label)
    assert bright_image.shape == image.shape
    assert bright_label.shape == label.shape
    assert torch.all(bright_image >= 0) and torch.all(bright_image <= 1)


def test_random_contrast(sample_image_label):
    image, label = sample_image_label
    contrast_image, contrast_label = random_contrast(image, label)
    assert contrast_image.shape == image.shape
    assert contrast_label.shape == label.shape
    assert torch.all(contrast_image >= 0) and torch.all(contrast_image <= 1)


def test_random_multiplicative_noise(sample_image_label):
    image, label = sample_image_label
    noisy_image, noisy_label = random_multiplicative_noise(image, label)
    assert noisy_image.shape == image.shape
    assert noisy_label.shape == label.shape
    assert torch.all(noisy_image >= 0) and torch.all(noisy_image <= 1)


def test_random_gamma(sample_image_label):
    image, label = sample_image_label
    gamma_image, gamma_label = random_gamma(image, label)
    assert gamma_image.shape == image.shape
    assert gamma_label.shape == label.shape
    assert torch.all(gamma_image >= 0) and torch.all(gamma_image <= 1)


def test_apply_augmentations(sample_image_label):
    image, label = sample_image_label
    aug_image, aug_label = apply_augmentations(image, label)
    assert aug_image.shape == image.shape
    assert aug_label.shape == label.shape
    assert torch.all(aug_image >= 0) and torch.all(aug_image <= 1)


def test_augmentations_class(sample_image_label):
    image, label = sample_image_label
    augmentations = Augmentations()
    aug_image, aug_label = augmentations(image, label)
    assert aug_image.shape == image.shape
    assert aug_label.shape == label.shape
    assert torch.all(aug_image >= 0) and torch.all(aug_image <= 1)


if __name__ == "__main__":
    pytest.main([__file__])
