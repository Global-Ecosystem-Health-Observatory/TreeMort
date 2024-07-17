import os
import argparse

from treemort.utils.config import setup

from treemort.data.dataset import prepare_datasets
from treemort.data.imagepaths import get_image_label_paths

from treemort.modeling.builder import resume_or_load
from treemort.modeling.trainer import trainer
from treemort.evaluation.evaluator import evaluator


def run(conf, eval_only):
    assert os.path.exists(
        conf.data_folder
    ), f"Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
            os.makedirs(conf.output_dir)

    train_images, train_labels, test_images, test_labels = get_image_label_paths(
        conf.data_folder
    )

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_images, train_labels, test_images, test_labels, conf
    )

    model = resume_or_load(conf)

    if eval_only:
        print("Evaluation-only mode started.")

        evaluator(model, test_dataset, len(test_images), conf.test_batch_size, conf.threshold)

    else:
        print("Training mode started.")
        
        trainer(model, train_dataset, val_dataset, len(train_images), conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for TreeMort network.")
    parser.add_argument("config",       type=str,                   help="Path to the configuration file")
    parser.add_argument("--eval-only",  action="store_true",        help="If set, only evaluate the model without training")
    
    args = parser.parse_args()

    conf = setup(args.config)
    
    run(conf, args.eval_only)
