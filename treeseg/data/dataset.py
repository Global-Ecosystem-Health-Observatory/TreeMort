import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from treeseg.utils.preprocess import load_and_crop_image, preprocess_image
from treeseg.utils.augment import CustomAugmentation

class TreeSegDataset(Dataset):
    def __init__(self, image_paths, label_paths, crop_size, input_channels, augment=False, binarize=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.crop_size = crop_size
        self.input_channels = input_channels
        self.augment = augment
        self.binarize = binarize
        self.augmentation_layer = CustomAugmentation() if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image, label = load_and_crop_image(image_path, label_path, self.crop_size, self.input_channels)
        image, label = preprocess_image(image, label, self.binarize)
        
        if self.augment:
            image, label = self.augmentation_layer(image, label)
        
        # Convert image to tensor
        # image = transforms.ToTensor()(image)

        # Convert label to tensor
        #label = torch.from_numpy(label).unsqueeze(0).float()  # Ensure label is float tensor with channel dimension
        
        # Ensure label has shape (H, W, 1)
        if label.ndim == 2:
            label = label.unsqueeze(-1)
        
        return image, label

def prepare_dataset(image_paths, label_paths, crop_size, batch_size, input_channels, augment=False, binarize=False, val_split_ratio=0.2):
    dataset = TreeSegDataset(image_paths, label_paths, crop_size, input_channels, augment, binarize)
    
    if val_split_ratio > 0:
        val_size = int(val_split_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    return train_loader, val_loader

def prepare_datasets(train_images, train_labels, test_images, test_labels, conf):
    train_loader, val_loader = prepare_dataset(
        train_images,
        train_labels,
        conf.train_crop_size,
        conf.train_batch_size,
        conf.input_channels,
        augment=True,
        binarize=conf.binarize,
        val_split_ratio=conf.val_size,
    )

    test_loader = prepare_dataset(
        test_images,
        test_labels,
        conf.test_crop_size,
        conf.test_batch_size,
        conf.input_channels,
        augment=False,
        binarize=conf.binarize,
        val_split_ratio=0,
    )[0]

    return train_loader, val_loader, test_loader


'''
#import tensorflow as tf

from treeseg.utils.preprocess import load_and_crop_image, preprocess_image
from treeseg.utils.augment import CustomAugmentation


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
'''