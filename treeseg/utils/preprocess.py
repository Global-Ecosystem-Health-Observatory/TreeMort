import numpy as np
import tensorflow as tf


def load_numpy(image_path, label_path, crop_size):
    image_np = np.load(image_path)
    label_np = np.load(label_path)

    label_np = np.expand_dims(label_np, axis=-1)

    concat = np.concatenate([image_np, label_np], axis=-1)

    row_pad_needed = max(0, crop_size - concat.shape[0])
    col_pad_needed = max(0, crop_size - concat.shape[1])

    padded_concat = np.pad(
        concat,
        [(0, row_pad_needed), (0, col_pad_needed), (0, 0)],
        mode="constant",
        constant_values=0,
    )

    image_np = padded_concat[:, :, :-1]
    label_np = padded_concat[:, :, -1]

    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    label_tf = tf.convert_to_tensor(label_np, dtype=tf.float32)

    return image_tf, label_tf


def normalize_inputs(image_tf, label_tf):
    image_tf = image_tf / 255.0

    label_tf = tf.cast(label_tf > 0, tf.float32)

    return image_tf, label_tf


def random_crop_tf(image_tf, label_tf, crop_size, image_channels):
    concat = tf.concat([image_tf, tf.expand_dims(label_tf, axis=-1)], axis=-1)

    cropped_concat = tf.image.random_crop(
        concat, size=[crop_size, crop_size, image_channels + 1]
    )

    cropped_image_tf = cropped_concat[:, :, :-1]
    cropped_label_index_tf = cropped_concat[:, :, -1]

    return cropped_image_tf, cropped_label_index_tf


def load_and_crop_image(image_path, label_path, crop_size, input_channels):
    image, label = tf.numpy_function(
        func=load_numpy,
        inp=[image_path, label_path, crop_size],
        Tout=[tf.float32, tf.float32],
    )

    image.set_shape([None, None, input_channels])
    label.set_shape([None, None])

    image, label = random_crop_tf(image, label, crop_size, input_channels)

    return image, label


def preprocess_image(image, label, crop_size):
    image, label = normalize_inputs(image, label)

    return image, label
