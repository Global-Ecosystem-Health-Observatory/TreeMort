import numpy as np
import torch
import torch.nn.functional as F


def load_numpy(image_path, label_path, crop_size):
    image_np = np.load(image_path)
    label_np = np.load(label_path)

    # Ensure label_np has shape (H, W, 1)
    if label_np.ndim == 2:
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
    label_np = padded_concat[:, :, -1:]

    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    label_tensor = torch.tensor(label_np, dtype=torch.float32)

    return image_tensor, label_tensor


def normalize_inputs(image_tensor, label_tensor):
    image_tensor = (image_tensor / 127.5) - 1.0  # rescale to range [-1, +1]
    label_tensor = label_tensor / 255.0  # rescale to range [0, +1]
    label_tensor = (
        label_tensor - (label_tensor == 0).float()
    )  # set 0 (background) to -1

    return image_tensor, label_tensor


def normalize_inputs_bin(image_tensor, label_tensor):
    image_tensor = image_tensor / 255.0
    label_tensor = (label_tensor > 0).float()

    return image_tensor, label_tensor


def random_crop(image_tensor, label_tensor, crop_size, image_channels):
    # Ensure that the label tensor has the same height and width as the image tensor
    if image_tensor.shape[:2] != label_tensor.shape[:2]:
        raise ValueError(
            f"Image shape {image_tensor.shape[:2]} and label shape {label_tensor.shape[:2]} do not match."
        )

    concat = torch.cat([image_tensor, label_tensor], dim=-1)

    if concat.shape[0] < crop_size or concat.shape[1] < crop_size:
        pad_height = max(0, crop_size - concat.shape[0])
        pad_width = max(0, crop_size - concat.shape[1])
        concat = F.pad(concat, (0, 0, 0, pad_width, 0, pad_height))

    height, width, _ = concat.shape
    top = torch.randint(0, height - crop_size + 1, (1,)).item()
    left = torch.randint(0, width - crop_size + 1, (1,)).item()

    cropped_concat = concat[top : top + crop_size, left : left + crop_size, :]
    cropped_image_tensor = cropped_concat[:, :, :image_channels]
    cropped_label_tensor = cropped_concat[:, :, image_channels:]

    return cropped_image_tensor, cropped_label_tensor


def load_and_crop_image(image_path, label_path, crop_size, input_channels):
    image, label = load_numpy(image_path, label_path, crop_size)
    image, label = random_crop(image, label, crop_size, input_channels)

    return image, label


def preprocess_image(image, label, binarize=False):
    if binarize:
        image, label = normalize_inputs_bin(image, label)
    else:
        image, label = normalize_inputs(image, label)

    return image, label


"""
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
    image_tf = (image_tf / 127.5) - 1.0  # rescale to range [-1, +1]

    label_tf = label_tf / 255.0  # rescale to range [0, +1]
    label_tf = label_tf - tf.cast(label_tf == 0, tf.float32)  # set 0 (background) to -1

    return image_tf, label_tf


def normalize_inputs_bin(image_tf, label_tf):
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


def preprocess_image(image, label, binarize=False):
    if binarize:
        image, label = normalize_inputs_bin(image, label)
    else:
        image, label = normalize_inputs(image, label)

    return image, label
"""
