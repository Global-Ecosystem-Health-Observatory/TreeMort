import tensorflow as tf

from treemort.utils.preprocess import load_and_crop_image, preprocess_image
from treemort.utils.augment import CustomAugmentation


def prepare_dataset(
    image_paths,
    label_paths,
    crop_size,
    batch_size,
    input_channels,
    augment=False,
    binarize=False,
    val_split_ratio=0.2,
):
    no_of_files = len(image_paths)
    dataset_paths = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    dataset_paths = dataset_paths.shuffle(
        buffer_size=no_of_files, reshuffle_each_iteration=True
    )

    if val_split_ratio > 0:
        val_size = int(val_split_ratio * no_of_files)
        val_dataset_paths = dataset_paths.take(val_size)
        train_dataset_paths = dataset_paths.skip(val_size)
    else:
        train_dataset_paths = dataset_paths
        val_dataset_paths = None

    def load_and_crop(image_path, label_path):
        return load_and_crop_image(image_path, label_path, crop_size, input_channels)

    def preprocess(image_path, label_path):
        return preprocess_image(image_path, label_path, binarize)

    augmentation_layer = CustomAugmentation() if augment else None

    def apply_augmentation(image, label):
        if augment:
            image, label = augmentation_layer(image, label)
        return image, label

    train_dataset = (
        train_dataset_paths.map(load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)
        .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    if val_dataset_paths:
        val_dataset = (
            val_dataset_paths.map(load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset


def prepare_datasets(train_images, train_labels, test_images, test_labels, conf):

    train_dataset, val_dataset = prepare_dataset(
        train_images,
        train_labels,
        conf.train_crop_size,
        conf.train_batch_size,
        conf.input_channels,
        augment=True,
        binarize=conf.binarize,
        val_split_ratio=conf.val_size,
    )

    test_dataset = prepare_dataset(
        test_images,
        test_labels,
        conf.test_crop_size,
        conf.test_batch_size,
        conf.input_channels,
        augment=False,
        binarize=conf.binarize,
        val_split_ratio=0,
    )[0]

    return train_dataset, val_dataset, test_dataset
