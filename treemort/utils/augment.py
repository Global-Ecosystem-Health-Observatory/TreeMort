import tensorflow as tf


class CustomAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomAugmentation, self).__init__()

    def call(self, image, label):
        combined = tf.concat([image, label], axis=-1)
        combined = self.random_flip(combined)
        combined = self.random_rotation(combined)

        image = combined[:, :, : image.shape[-1]]
        label = combined[:, :, image.shape[-1] :]

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
