import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from treeseg.utils.augment import Augmentations


class DeadTreeDataset(Dataset):
    def __init__(
        self, root_dir, split="train", crop_size=256, transform=None, binarize=False
    ):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.transform = transform
        self.binarize = binarize
        self.images_dir = os.path.join(root_dir, split, "Images")
        self.labels_dir = os.path.join(root_dir, split, "Labels")
        self.image_files = [
            f for f in os.listdir(self.images_dir) if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx])

        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        if self.binarize:
            image = image / 255.0
            label = (label > 0).astype(np.float32)

        else:
            pass

        image, label = self.center_crop_or_pad(image, label, self.crop_size)

        if self.transform:
            image, label = self.transform(image, label)

        # Convert to torch tensors
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W) format
        label = torch.tensor(label).unsqueeze(0)

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


def prepare_datasets(root_dir, conf):

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    full_train_dataset = DeadTreeDataset(
        root_dir,
        split="Train",
        crop_size=conf.train_crop_size,
        transform=train_transform,
        binarize=conf.binarize,
    )
    test_dataset = DeadTreeDataset(
        root_dir,
        split="Test",
        crop_size=conf.test_crop_size,
        transform=test_transform,
        binarize=conf.binarize,
    )

    val_size = int(conf.val_size * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=conf.train_batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=conf.train_batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=conf.test_batch_size, shuffle=False, drop_last=True
    )

    return train_loader, val_loader, test_loader
