import pytest
import torch

from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

from treemort.utils.metrics import iou_score


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def sample_data():
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_model.to(device)
    
    simple_model.eval()
    
    images, labels = next(iter(sample_data))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = simple_model(images)

    result = iou_score(preds, labels, threshold=0.5)

    assert 0 <= result <= 1


if __name__ == "__main__":
    pytest.main([__file__])
