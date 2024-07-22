import os
import torch
import shutil
import pytest
import tempfile

import numpy as np

from torch.utils.data import DataLoader
from treemort.data.dataset import DeadTreeDataset, prepare_datasets


# Configuration class to simulate configuration object passed to prepare_datasets
class Config:
    def __init__(self):
        self.train_crop_size = 256
        self.test_crop_size = 256
        self.binarize = True
        self.val_size = 0.2
        self.train_batch_size = 2
        self.test_batch_size = 2


# Helper function to create a mock dataset
def create_mock_dataset(root_dir):
    splits = ["Train", "Test"]
    for split in splits:
        images_dir = os.path.join(root_dir, split, "Images")
        labels_dir = os.path.join(root_dir, split, "Labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        for i in range(5):  # Create 5 mock files for each split
            image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
            label = np.random.randint(0, 2, (300, 300), dtype=np.uint8)
            np.save(os.path.join(images_dir, f"{i}.npy"), image)
            np.save(os.path.join(labels_dir, f"{i}.npy"), label)


@pytest.fixture(scope="module")
def setup_mock_dataset():
    test_dir = tempfile.mkdtemp()
    create_mock_dataset(test_dir)
    config = Config()
    yield test_dir, config
    shutil.rmtree(test_dir)


def test_dataset_length(setup_mock_dataset):
    test_dir, _ = setup_mock_dataset
    dataset = DeadTreeDataset(root_dir=test_dir, split="Train", crop_size=256)
    assert len(dataset) == 5


def test_dataset_item(setup_mock_dataset):
    test_dir, _ = setup_mock_dataset
    dataset = DeadTreeDataset(
        root_dir=test_dir, split="Train", crop_size=256, binarize=True
    )
    image, label = dataset[0]
    assert image.shape == (3, 256, 256)
    assert label.shape == (1, 256, 256)
    assert torch.is_tensor(image)
    assert torch.is_tensor(label)


def test_data_loaders(setup_mock_dataset):
    test_dir, config = setup_mock_dataset
    train_loader, val_loader, test_loader = prepare_datasets(test_dir, config)

    # Check if loaders have the correct length
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 5

    # Check if loaders yield the correct batch size
    train_batch = next(iter(train_loader))
    assert train_batch[0].shape == (config.train_batch_size, 3, 256, 256)
    assert train_batch[1].shape == (config.train_batch_size, 1, 256, 256)

    test_batch = next(iter(test_loader))
    assert test_batch[0].shape == (config.test_batch_size, 3, 256, 256)
    assert test_batch[1].shape == (config.test_batch_size, 1, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__])
