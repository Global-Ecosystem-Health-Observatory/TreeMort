import os
import torch

import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, MaskFormerImageProcessor

from treemort.utils.augment import Augmentations


class DeadTreeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        crop_size=256,
        transform=None,
        binarize=False,
        image_processor=None,
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
        self.image_processor = image_processor

        # Ensure image_mean and image_std have four values
        if image_processor:
            if len(self.image_processor.image_mean) == 3:
                self.image_processor.image_mean.append(0.5)
            if len(self.image_processor.image_std) == 3:
                self.image_processor.image_std.append(0.5)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx])

        image = torch.from_numpy(np.load(image_path).astype(np.float32))
        label = torch.from_numpy(np.load(label_path).astype(np.float32))

        if self.binarize:
            image = image / 255.0
            label = (label > 0).float()

        image = image.permute(2, 0, 1)  # Convert to (C, H, W) format

        image, label = self.center_crop_or_pad(image, label, self.crop_size)

        if self.image_processor:
            image = F.resize(
                image,
                [
                    self.image_processor.size["height"],
                    self.image_processor.size["width"],
                ],
            )
            image = (
                image - torch.tensor(self.image_processor.image_mean).view(-1, 1, 1)
            ) / torch.tensor(self.image_processor.image_std).view(-1, 1, 1)

        if self.transform:
            image, label = self.transform(image, label)

        label = label.unsqueeze(0)  # Convert to (1, H, W) format

        return image, label


    def center_crop_or_pad(self, image, label, size=256):
        h, w = image.shape[1:]  # image is in (C, H, W) format
        ch, cw = (size, size)

        # Padding if image dimensions are smaller than crop size
        if h < ch or w < cw:
            pad_h = max(ch - h, 0)
            pad_w = max(cw - w, 0)
            image = torch.nn.functional.pad(
                image,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="constant",
                value=0,
            )
            label = torch.nn.functional.pad(
                label,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="constant",
                value=0,
            )
            h, w = image.shape[1:]

        # Cropping
        x = (w - cw) // 2
        y = (h - ch) // 2
        image = image[:, y : y + ch, x : x + cw]
        label = label[y : y + ch, x : x + cw]

        return image, label


def prepare_datasets(root_dir, conf):

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    if conf.model == "maskformer":
        image_processor = MaskFormerImageProcessor(
            ignore_index=0, do_resize=True, do_rescale=False, do_normalize=False
        )
    elif conf.model == "detr":
        image_processor = AutoImageProcessor.from_pretrained(
            "facebook/detr-resnet-50-panoptic"
        )
    elif conf.model == "beit":
        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640", do_rescale=False
        )
        image_processor.size = (384,384) # default (640,640) exceeds GPU memory

    else:
        image_processor = None

    full_train_dataset = DeadTreeDataset(
        root_dir,
        split="Train",
        crop_size=conf.train_crop_size,
        transform=train_transform,
        binarize=conf.binarize,
        image_processor=image_processor,
    )
    test_dataset = DeadTreeDataset(
        root_dir,
        split="Test",
        crop_size=conf.test_crop_size,
        transform=test_transform,
        binarize=conf.binarize,
        image_processor=image_processor,
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

    return train_loader, val_loader, test_loader, image_processor
