import numpy as np
import tensorflow as tf

from unittest.mock import patch

from treemort.data.dataset import prepare_dataset


def test_prepare_dataset():
    crop_size = 256
    batch_size = 8
    input_channels = 4
    num_samples = 100

    image_paths = [f"image_{i}.npy".encode("utf-8") for i in range(num_samples)]
    label_paths = [f"label_{i}.npy".encode("utf-8") for i in range(num_samples)]

    # Mock np.load to return random data
    def mock_np_load(path):
        if b"image" in path:
            return np.random.randint(
                low=0, high=255, size=(crop_size, crop_size, input_channels)
            ).astype(np.float32)
        elif b"label" in path:
            return 1.0 - np.random.power(1.0, (crop_size, crop_size))

    with patch("numpy.load", side_effect=mock_np_load):
        try:
            train_dataset, val_dataset = prepare_dataset(
                image_paths,
                label_paths,
                crop_size,
                batch_size,
                input_channels,
                augment=False,
            )

            assert len(train_dataset) > 0, "Train dataset is empty"
            assert len(val_dataset) > 0, "Validation dataset is empty"

            for images, labels in train_dataset.take(1):
                assert images.shape == (
                    batch_size,
                    crop_size,
                    crop_size,
                    input_channels,
                ), "Image batch shape is incorrect"
                assert labels.shape == (
                    batch_size,
                    crop_size,
                    crop_size,
                ), "Label batch shape is incorrect"

            print("prepare_dataset test passed")
        except Exception as e:
            print(f"prepare_dataset test failed: {e}")
