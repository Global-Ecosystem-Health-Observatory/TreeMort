import os
import h5py

import tensorflow as tf

from treemort.utils.preprocess import preprocess_image
from treemort.utils.datautils import stratified_split
from treemort.utils.augment import CustomAugmentation


def load_from_hdf5(hdf5_file_path, key):
    with h5py.File(hdf5_file_path, "r") as hf:
        image = hf[key]["image"][()]
        label = hf[key]["label"][()]
    return image, label


def decode_and_reshape(image, label, crop_size, input_channels, output_channels):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)

    image = tf.reshape(image, [crop_size, crop_size, input_channels])
    label = tf.reshape(label, [crop_size, crop_size, output_channels])

    return image, label


def load_and_decode(hdf5_file_path, key):
    key = key.numpy().decode("utf-8")  # Convert bytes to string
    image, label = load_from_hdf5(hdf5_file_path, key)
    return image, label


def prepare_dataset(
    hdf5_file_path,
    keys,
    crop_size,
    batch_size,
    input_channels,
    output_channels,
    augment=False,
):
    def load_and_decode_fn(key):
        key = tf.convert_to_tensor(key, dtype=tf.string)
        image, label = tf.py_function(
            func=lambda k: load_and_decode(hdf5_file_path, k),
            inp=[key],
            Tout=[tf.float32, tf.float32],
        )

        label = tf.expand_dims(label, axis=-1)

        image.set_shape([crop_size, crop_size, input_channels])
        label.set_shape([crop_size, crop_size, output_channels])
        return image, label

    def decode_and_reshape_fn(image, label):
        return decode_and_reshape(image, label, crop_size, input_channels, output_channels)

    def apply_augmentation(image, label):
        if augment:
            augmentation_layer = CustomAugmentation()
            image, label = augmentation_layer(image, label)
        return image, label

    def preprocess_fn(image, label):
        image, label = preprocess_image(image, label)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(keys)
    dataset = dataset.map(lambda key: load_and_decode_fn(key), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(decode_and_reshape_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_datasets(conf):
    hdf5_file_path = os.path.join(conf.data_folder, conf.hdf5_file)

    train_keys, val_keys, test_keys = stratified_split(hdf5_file_path, conf.val_size, conf.test_size)

    train_dataset = prepare_dataset(
        hdf5_file_path,
        train_keys,
        conf.train_crop_size,
        conf.train_batch_size,
        conf.input_channels,
        conf.output_channels,
        augment=True,
    )

    val_dataset = prepare_dataset(
        hdf5_file_path,
        val_keys,
        conf.val_crop_size,
        conf.val_batch_size,
        conf.input_channels,
        conf.output_channels,
        augment=False,
    )

    test_dataset = prepare_dataset(
        hdf5_file_path,
        test_keys,
        conf.test_crop_size,
        conf.test_batch_size,
        conf.input_channels,
        conf.output_channels,
        augment=False,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        len(train_keys),
        len(val_keys),
        len(test_keys),
    )
