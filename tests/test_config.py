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
        --hdf5-file = dataset.h5
        --model resnet
        --epochs 10
        --train-batch-size 8
        --val-batch-size 8
        --test-batch-size 8
        --train-crop-size 256
        --val-crop-size 256
        --test-crop-size 256
        --input-channels 3
        --output-channels 1
        """
        )
        return f.name


def test_setup(temp_config_file):
    conf = setup(temp_config_file)

    print(conf)

    assert conf.data_folder == "/path/to/data"
    assert conf.model == "resnet"
    assert conf.epochs == 10
    assert conf.train_batch_size == 8
    assert conf.val_batch_size == 8
    assert conf.test_batch_size == 8
    assert conf.train_crop_size == 256
    assert conf.val_crop_size == 256
    assert conf.test_crop_size == 256
    assert conf.input_channels == 3
    assert conf.output_channels == 1
    assert conf.val_size == 0.2
    assert conf.test_size == 0.1
    assert conf.output_dir == "output"
    assert conf.model_weights == "latest"
    assert conf.learning_rate == 2e-4
    assert conf.segment_threshold == 0.5
    assert conf.activation == "sigmoid"
    assert conf.loss == "hybrid"
    assert not conf.resume


if __name__ == "__main__":
    pytest.main([__file__])
