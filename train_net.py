import os
import argparse

from tools.config import setup

from treeseg.data.dataset import prepare_datasets
from treeseg.data.imagepaths import get_image_label_paths

from treeseg.modeling.trainer import resume_or_load, trainer
from treeseg.evaluation.evaluator import evaluator

def run(conf, eval_only):
    assert os.path.exists(
        conf.data_folder
    ), f"Data folder {conf.data_folder} does not exist."

    train_images, train_labels, test_images, test_labels = get_image_label_paths(
        conf.data_folder
    )

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_images, train_labels, test_images, test_labels, conf
    )

    model = resume_or_load(conf)

    if eval_only:
        print("Evaluation only mode")
        evaluator(model, test_dataset, len(test_images), conf.test_batch_size)

    else:
        print("Training mode")
        trainer(model, train_dataset, val_dataset, len(train_images), conf)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, only evaluate the model without training",
    )

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)
