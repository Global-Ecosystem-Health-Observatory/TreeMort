import os
import argparse

from tools.config import setup
from treeseg.data.dataset import prepare_datasets
from treeseg.modeling.trainer import trainer
from treeseg.modeling.builder import resume_or_load
from treeseg.evaluation.evaluator import evaluator
from treeseg.utils.callbacks import build_callbacks


def run(conf, eval_only):
    assert os.path.exists(
        conf.data_folder
    ), f"Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)

    train_dataset, val_dataset, test_dataset = prepare_datasets(conf.data_folder, conf)

    model, optimizer, criterion, metrics = resume_or_load(conf)

    n_batches = len(train_dataset)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if eval_only:
        print("Evaluation-only mode started.")

        evaluator(
            model, test_dataset, len(test_dataset), conf.test_batch_size, conf.threshold
        )

    else:
        print("Training mode started.")

        trainer(
            model,
            optimizer,
            criterion,
            metrics,
            train_dataset,
            val_dataset,
            conf,
            callbacks,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config",       type=str,               help="Path to the configuration file")
    parser.add_argument("--eval-only",  action="store_true",    help="If set, only evaluate the model without training",)

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)
