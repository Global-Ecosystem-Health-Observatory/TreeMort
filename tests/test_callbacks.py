import os
import torch
import pytest

from torch import nn, optim
from unittest.mock import patch
from pyfakefs.fake_filesystem_unittest import Patcher

from treemort.utils.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from treemort.modeling.callback_builder import build_callbacks


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.Adam(dummy_model.parameters(), lr=0.001)


@patch("torch.save")
def test_model_checkpoint(mock_torch_save, dummy_model, dummy_optimizer):
    with Patcher() as patcher:
        fake_output_dir = "/fake_output_dir"
        fake_checkpoints_dir = os.path.join(fake_output_dir, "Checkpoints")
        checkpoint_path = os.path.join(
            fake_checkpoints_dir, "cp-{epoch:04d}.weights.pth"
        )

        patcher.fs.create_dir(fake_output_dir)
        patcher.fs.create_dir(fake_checkpoints_dir)

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_freq=1)

        model_checkpoint(epoch=1, model=dummy_model, optimizer=dummy_optimizer)

        assert mock_torch_save.called

        saved_model_state = mock_torch_save.call_args[0][0]
        expected_model_state = dummy_model.state_dict()

        for key in expected_model_state:
            assert torch.equal(expected_model_state[key], saved_model_state[key])


@patch("torch.save")
def test_model_checkpoint_best(mock_torch_save, dummy_model, dummy_optimizer):
    with Patcher() as patcher:
        fake_output_dir = "/fake_output_dir"
        best_checkpoint_path = os.path.join(fake_output_dir, "best.weights.pth")

        patcher.fs.create_dir(fake_output_dir)

        model_checkpoint = ModelCheckpoint(
            filepath=best_checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

        # Simulate first call with a validation loss
        val_loss = 0.5
        model_checkpoint(epoch=1, model=dummy_model, optimizer=dummy_optimizer, val_loss=val_loss)

        assert mock_torch_save.called

        saved_model_state = mock_torch_save.call_args[0][0]
        expected_model_state = dummy_model.state_dict()

        for key in expected_model_state:
            assert torch.equal(expected_model_state[key], saved_model_state[key])


@patch("torch.save")
def test_reduce_lr_on_plateau(mock_torch_save, dummy_optimizer):
    with Patcher() as patcher:
        fake_output_dir = "/fake_output_dir"
        patcher.fs.create_dir(fake_output_dir)

        reduce_lr = ReduceLROnPlateau(optimizer=dummy_optimizer, patience=1, verbose=0)

        reduce_lr(0.5)  # Pass the monitored value directly
        initial_lr = dummy_optimizer.param_groups[0]["lr"]

        reduce_lr(0.6)  # Pass the monitored value directly
        reduced_lr = dummy_optimizer.param_groups[0]["lr"]
        assert reduced_lr < initial_lr


@patch("torch.save")
def test_early_stopping(mock_torch_save):
    with Patcher() as patcher:
        fake_output_dir = "/fake_output_dir"
        patcher.fs.create_dir(fake_output_dir)

        early_stopping = EarlyStopping(patience=1, verbose=0)

        early_stopping(1, 0.5)
        assert not early_stopping.stop_training

        early_stopping(2, 0.6)
        assert early_stopping.stop_training


@patch("torch.save")
def test_build_callbacks(mock_torch_save, dummy_optimizer):
    with Patcher() as patcher:
        fake_output_dir = "/fake_output_dir"
        fake_checkpoints_dir = os.path.join(fake_output_dir, "Checkpoints")

        patcher.fs.create_dir(fake_output_dir)
        patcher.fs.create_dir(fake_checkpoints_dir)

        callbacks = build_callbacks(
            n_batches=100, output_dir=fake_output_dir, optimizer=dummy_optimizer
        )

        assert isinstance(callbacks, list)
        # assert len(callbacks) == 3
        assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
        # assert any(isinstance(cb, ReduceLROnPlateau) for cb in callbacks)
        assert any(isinstance(cb, EarlyStopping) for cb in callbacks)


if __name__ == "__main__":
    pytest.main([__file__])
