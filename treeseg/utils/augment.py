import torch
import torchvision.transforms.functional as F
import random

class CustomAugmentation:
    def __init__(self):
        pass

    def __call__(self, image, label):
        # Ensure label has shape (H, W, 1)
        if label.ndim == 2:
            label = label.unsqueeze(-1)

        # Concatenate image and label along the last dimension
        combined = torch.cat((image, label.float()), dim=-1)

        # Apply augmentations that affect both image and label
        combined = self.random_flip(combined)
        combined = self.random_rotation(combined)

        # Split back into image and label tensors
        image = combined[..., :image.shape[-1]]  # Assuming image has shape (H, W, 4)
        label = combined[..., image.shape[-1]:].squeeze(-1)  # Assuming label has shape (H, W, 1)

        # Apply augmentations that affect only image
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_multiplicative_noise(image)
        image = self.random_gamma(image)

        return image, label

    def random_flip(self, combined):
        if random.random() > 0.5:
            combined = torch.flip(combined, dims=[1])
        if random.random() > 0.5:
            combined = torch.flip(combined, dims=[2])
        return combined

    def random_rotation(self, combined):
        k = random.randint(0, 3)
        combined = torch.rot90(combined, k, dims=[1, 2])
        return combined

    def random_brightness(self, image):
        brightness_factor = random.uniform(0.8, 1.2)

        # Apply brightness adjustment channel-wise
        for c in range(image.shape[-1]):
            image[..., c] = self.adjust_brightness(image[..., c], brightness_factor)

        return image

    def adjust_brightness(self, img, brightness_factor):
        # Custom implementation of brightness adjustment
        return torch.clamp(img * brightness_factor, 0, 1)

    def random_contrast(self, image):
        contrast_factor = random.uniform(0.8, 1.2)

        # Apply contrast adjustment channel-wise
        for c in range(image.shape[-1]):
            image[..., c] = self.adjust_contrast(image[..., c], contrast_factor)

        return image

    def adjust_contrast(self, img, contrast_factor):
        # Custom implementation of contrast adjustment
        mean = img.mean(dim=[-2, -1], keepdim=True)
        return torch.clamp((img - mean) * contrast_factor + mean, 0, 1)

    def random_multiplicative_noise(self, image):
        noise = torch.empty_like(image).uniform_(0.8, 1.2)
        return image * noise

    def random_gamma(self, image):
        gamma = random.uniform(0.8, 1.2)
        image = torch.pow(image, gamma)
        image = torch.clamp(image, 0, 1)
        return image


'''
import tensorflow as tf


class CustomAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomAugmentation, self).__init__()

    def call(self, image, label):
        combined = tf.concat([image, tf.expand_dims(label, axis=-1)], axis=-1)
        combined = self.random_flip(combined)
        combined = self.random_rotation(combined)

        image = combined[:, :, : image.shape[-1]]
        label = tf.squeeze(combined[:, :, image.shape[-1] :], axis=-1)

        image = self.random_brightness_contrast(image)
        image = self.random_multiplicative_noise(image)
        image = self.random_gamma(image)

        return image, label

    def random_flip(self, combined):
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        return combined

    def random_rotation(self, combined):
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k)
        return combined

    def random_brightness_contrast(self, image):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image

    def random_multiplicative_noise(self, image):
        multiplier = tf.random.uniform(tf.shape(image), 0.8, 1.2)
        image = image * multiplier
        return image

    def random_gamma(self, image):
        gamma = tf.random.uniform([], 0.8, 1.2)
        image = tf.image.adjust_gamma(image, gamma=gamma)
        image = tf.clip_by_value(image, 0.0, 255.0)

        # Handle NaNs by replacing them with zero
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)

        return image
'''