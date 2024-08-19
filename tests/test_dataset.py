import os
import h5py
import pytest
import tempfile
import shutil

import numpy as np

from treemort.data.dataset import DeadTreeDataset
from treemort.data.loader import prepare_datasets


# Mock configuration class
class DataLoaderConfig:
    def __init__(self, data_folder, hdf5_file):
        self.data_folder = data_folder
        self.hdf5_file = hdf5_file
        self.val_size = 0.2
        self.test_size = 0.1
        self.train_crop_size = 256
        self.val_crop_size = 256
        self.test_crop_size = 256
        self.train_batch_size = 2
        self.val_batch_size = 2
        self.test_batch_size = 2
        self.model = "sa_unet"  # For consistency with your config


def create_mock_hdf5_dataset(hdf5_file):
    with h5py.File(hdf5_file, "w") as hf:
        for i in range(20):  # Create 20 mock data entries
            file_stub = f"image_{i}"
            for idx in range(2):  # Each image has two patches
                key = f"{file_stub}_{idx}"
                grp = hf.create_group(key)

                image_patch = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
                label_patch = np.random.randint(0, 2, (256, 256), dtype=np.uint8)

                grp.create_dataset("image", data=image_patch, compression="gzip")
                grp.create_dataset("label", data=label_patch, compression="gzip")
                grp.attrs["contains_dead_tree"] = bool(np.random.randint(0, 2))
                grp.attrs["source_image"] = f"source_image_{i}.tif"


@pytest.fixture(scope="module")
def setup_mock_dataset():
    temp_dir = tempfile.mkdtemp()
    hdf5_file_path = os.path.join(temp_dir, "test_dataset.h5")

    create_mock_hdf5_dataset(hdf5_file_path)

    config = DataLoaderConfig(data_folder=temp_dir, hdf5_file="test_dataset.h5")

    yield config

    shutil.rmtree(temp_dir)  # Clean up the temporary directory


def test_data_loaders(setup_mock_dataset):
    config = setup_mock_dataset

    train_loader, val_loader, test_loader, image_processor = prepare_datasets(config)

    assert len(train_loader.dataset) > 0, "Train dataset is empty."
    assert len(val_loader.dataset) > 0, "Validation dataset is empty."
    assert len(test_loader.dataset) > 0, "Test dataset is empty."

    train_batch = next(iter(train_loader))
    assert train_batch[0].shape == (config.train_batch_size, 4, 256, 256)
    assert train_batch[1].shape == (config.train_batch_size, 1, 256, 256)

    test_batch = next(iter(test_loader))
    assert test_batch[0].shape == (config.test_batch_size, 4, 256, 256)
    assert test_batch[1].shape == (config.test_batch_size, 1, 256, 256)


if __name__ == "__main__":
    pytest.main([__file__])
