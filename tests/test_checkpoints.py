import os
import pytest

from pyfakefs.fake_filesystem_unittest import Patcher
from treemort.utils.checkpoints import get_checkpoint


def test_get_checkpoint_best():
    with Patcher() as patcher:
        patcher.fs.create_file("/fake_output_dir/best.weights.pth")

        output_dir = "/fake_output_dir"
        model_weights = "best"

        checkpoint = get_checkpoint(model_weights, output_dir)
        assert checkpoint == "/fake_output_dir/best.weights.pth"


def test_get_checkpoint_best_not_found():
    with Patcher() as patcher:
        output_dir = "/fake_output_dir"
        model_weights = "best"

        checkpoint = get_checkpoint(model_weights, output_dir)
        assert checkpoint is None


def test_get_checkpoint_latest():
    with Patcher() as patcher:
        patcher.fs.create_dir("/fake_output_dir/Checkpoints")
        patcher.fs.create_file("/fake_output_dir/Checkpoints/1.weights.pth")
        patcher.fs.create_file("/fake_output_dir/Checkpoints/2.weights.pth")

        output_dir = "/fake_output_dir"
        model_weights = "latest"

        checkpoint = get_checkpoint(model_weights, output_dir)
        assert checkpoint == "/fake_output_dir/Checkpoints/2.weights.pth"


def test_get_checkpoint_latest_no_files():
    with Patcher() as patcher:
        patcher.fs.create_dir("/fake_output_dir/Checkpoints")

        output_dir = "/fake_output_dir"
        model_weights = "latest"

        checkpoint = get_checkpoint(model_weights, output_dir)
        assert checkpoint is None


def test_get_checkpoint_latest_empty_directory():
    with Patcher() as patcher:
        patcher.fs.create_dir("/fake_output_dir/Checkpoints")

        output_dir = "/fake_output_dir"
        model_weights = "latest"

        checkpoint = get_checkpoint(model_weights, output_dir)
        assert checkpoint is None


# Add this if you want to run the tests without pytest command
if __name__ == "__main__":
    pytest.main([__file__])
