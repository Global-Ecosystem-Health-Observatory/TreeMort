import torch
import pytest
from treemort.utils.loss import (
    dice_loss,
    focal_loss,
    hybrid_loss,
    mse_loss,
    iou_score,
    f_score,
)


@pytest.fixture
def sample_data():
    pred = torch.tensor([[0.8, 0.2], [0.4, 0.9]], dtype=torch.float32)
    target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    return pred, target


def test_dice_loss(sample_data):
    pred, target = sample_data
    loss = dice_loss(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_focal_loss(sample_data):
    pred, target = sample_data
    loss = focal_loss(pred, target, alpha=0.8, gamma=2)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_hybrid_loss(sample_data):
    pred, target = sample_data
    loss = hybrid_loss(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_mse_loss(sample_data):
    pred, target = sample_data
    loss = mse_loss(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_iou_score(sample_data):
    pred, target = sample_data
    score = iou_score(pred, target, threshold=0.5)
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1


def test_f_score(sample_data):
    pred, target = sample_data
    score = f_score(pred, target, threshold=0.5, beta=1)
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1


if __name__ == "__main__":
    pytest.main([__file__])
