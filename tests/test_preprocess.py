import os

import numpy as np
import tensorflow as tf

from treemort.utils.preprocess import load_numpy, normalize_inputs, random_crop_tf


def test_load_numpy():
    
    image_data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    label_data = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)

    image_path = 'temp_image.npy'
    label_path = 'temp_label.npy'
    np.save(image_path, image_data)
    np.save(label_path, label_data)

    crop_size = 128

    image_tf, label_tf = load_numpy(image_path, label_path, crop_size)

    assert image_tf.shape == (crop_size, crop_size, 3), f"Expected shape: {(crop_size, crop_size, 3)}, but got: {image_tf.shape}"
    assert label_tf.shape == (crop_size, crop_size), f"Expected shape: {(crop_size, crop_size)}, but got: {label_tf.shape}"

    os.remove(image_path)
    os.remove(label_path)


def test_normalize_inputs():

    image_data = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    label_data = np.random.randint(0, 2, size=(128, 128), dtype=np.uint8)

    image_tf = tf.convert_to_tensor(image_data, dtype=tf.float32)
    label_tf = tf.convert_to_tensor(label_data, dtype=tf.float32)

    normalized_image_tf, normalized_label_tf = normalize_inputs(image_tf, label_tf)

    assert tf.reduce_min(normalized_image_tf) >= -1.0, f"Image min value out of range: {tf.reduce_min(normalized_image_tf).numpy()}"
    assert tf.reduce_max(normalized_image_tf) <= 1.0, f"Image max value out of range: {tf.reduce_max(normalized_image_tf).numpy()}"

    assert tf.reduce_min(normalized_label_tf) >= -1.0, f"Label min value out of range: {tf.reduce_min(normalized_label_tf).numpy()}"
    assert tf.reduce_max(normalized_label_tf) <= 1.0, f"Label max value out of range: {tf.reduce_max(normalized_label_tf).numpy()}"

def test_random_crop_tf():
    # Create sample image and label data
    height, width, image_channels = 256, 256, 3
    crop_size = 128

    image_np = np.random.randint(0, 256, size=(height, width, image_channels), dtype=np.uint8)
    label_index_np = np.random.randint(0, 2, size=(height, width), dtype=np.uint8)

    # Convert to TensorFlow tensors
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    label_index_tf = tf.convert_to_tensor(label_index_np, dtype=tf.float32)

    # Perform random crop
    cropped_image_tf, cropped_label_index_tf = random_crop_tf(image_tf, label_index_tf, crop_size, image_channels)

    # Assert the shapes
    assert cropped_image_tf.shape == (crop_size, crop_size, image_channels), \
        f"Expected cropped image shape: ({crop_size}, {crop_size}, {image_channels}), but got: {cropped_image_tf.shape}"
    assert cropped_label_index_tf.shape == (crop_size, crop_size), \
        f"Expected cropped label shape: ({crop_size}, {crop_size}), but got: {cropped_label_index_tf.shape}"

    # Check the values are within the expected range
    assert tf.reduce_min(cropped_image_tf).numpy() >= 0, \
        f"Cropped image has values less than 0: {tf.reduce_min(cropped_image_tf).numpy()}"
    assert tf.reduce_max(cropped_image_tf).numpy() <= 255, \
        f"Cropped image has values greater than 255: {tf.reduce_max(cropped_image_tf).numpy()}"

