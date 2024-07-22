import pytest
import tempfile
import configargparse

from treemort.utils.config import setup


# Create a temporary configuration file for testing
@pytest.fixture
def temp_config_file():
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write(
            """
        --data-folder /path/to/data
        --model resnet
        --epochs 10
        --train-batch-size 16
        --test-batch-size 8
        --train-crop-size 256
        --test-crop-size 256
        --input-channels 3
        --output-channels 1
        """
        )
        return f.name


def test_setup(temp_config_file):
    conf = setup(temp_config_file)

    assert conf.data_folder == "/path/to/data"
    assert conf.model == "resnet"
    assert conf.epochs == 10
    assert conf.train_batch_size == 16
    assert conf.test_batch_size == 8
    assert conf.train_crop_size == 256
    assert conf.test_crop_size == 256
    assert conf.input_channels == 3
    assert conf.output_channels == 1
    assert conf.val_size == 0.2
    assert conf.output_dir == "output"
    assert conf.model_weights == "latest"
    assert conf.learning_rate == 2e-4
    assert conf.threshold == -0.5
    assert conf.activation == "tanh"
    assert conf.loss == "mse"
    assert not conf.resume
    assert not conf.binarize


# Add this if you want to run the tests without pytest command
if __name__ == "__main__":
    pytest.main([__file__])
