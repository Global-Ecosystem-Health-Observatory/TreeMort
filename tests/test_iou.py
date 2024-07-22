import pytest
import torch

import numpy as np

from unittest.mock import MagicMock
from treemort.utils.iou import IOUCallback
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def sample_data():
    # Create sample data
    images = torch.rand(10, 1, 256, 256)  # 10 images of size 256x256 with 1 channel
    labels = torch.randint(
        0, 2, (10, 1, 256, 256), dtype=torch.float32
    )  # Binary labels
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    return dataloader


@pytest.fixture
def simple_model():
    model = SimpleModel()
    return model


def test_iou_callback(sample_data, simple_model):
    num_samples = 10
    batch_size = 2
    threshold = 0.5

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_model.to(device)

    # Create IOUCallback instance
    iou_callback = IOUCallback(
        model=simple_model,
        dataset=sample_data,
        num_samples=num_samples,
        batch_size=batch_size,
        threshold=threshold,
    )

    # Evaluate the model using IOUCallback
    result = iou_callback.evaluate()

    assert isinstance(result, dict)
    assert "mean_iou_pixels" in result
    assert "mean_iou_trees" in result
    assert 0 <= result["mean_iou_pixels"] <= 1
    assert 0 <= result["mean_iou_trees"] <= 1


# Add this if you want to run the tests without pytest command
if __name__ == "__main__":
    pytest.main([__file__])
