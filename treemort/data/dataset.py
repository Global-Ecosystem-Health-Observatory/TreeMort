import os
import h5py
import torch

import numpy as np
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader, random_split

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import stratified_split


class DeadTreeDataset(Dataset):
    def __init__(
        self, hdf5_file, keys, crop_size=256, transform=None
    ):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, "r") as hf:
            key = self.keys[idx]
            image = hf[key]['image'][()]
            label = hf[key]['label'][()]

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        image = image / 255.0
        # label = (label > 0).astype(np.float32) # binarized within hdf5 dataset

        image, label = self.center_crop_or_pad(image, label, self.crop_size)

        if self.transform:
            image, label = self.transform(image, label)

        # Convert to torch tensors
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W) format
        label = torch.tensor(label).unsqueeze(0)  # Add channel dimension for label

        return image, label

    def center_crop_or_pad(self, image, label, size=256):
        h, w = image.shape[:2]
        ch, cw = (size, size)

        # Padding if image dimensions are smaller than crop size
        if h < ch or w < cw:
            pad_h = max(ch - h, 0)
            pad_w = max(cw - w, 0)
            image = np.pad(
                image,
                (
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode="constant",
                constant_values=0,
            )
            h, w = image.shape[:2]

        # Cropping
        x = (w - cw) // 2
        y = (h - ch) // 2
        image = image[y : y + ch, x : x + cw]
        label = label[y : y + ch, x : x + cw]

        return image, label



def prepare_datasets(conf):

    hdf5_file_path = os.path.join(conf.data_folder, conf.hdf5_file)

    train_keys, val_keys, test_keys = stratified_split(hdf5_file_path, conf.val_size, conf.test_size)

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    train_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=train_keys,
        crop_size=conf.train_crop_size,
        transform=train_transform,
    )
    val_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=val_keys,
        crop_size=conf.val_crop_size,
        transform=val_transform,
    )
    test_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=test_keys,
        crop_size=conf.test_crop_size,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=conf.train_batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=conf.val_batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=conf.test_batch_size, shuffle=False, drop_last=True
    )

    return train_loader, val_loader, test_loader
