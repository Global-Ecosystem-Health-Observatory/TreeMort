import os
import torch
import argparse

from treemort.utils.config import setup
from treemort.utils.callbacks import build_callbacks

from treemort.data.dataset import prepare_datasets

from treemort.modeling.builder import resume_or_load
from treemort.modeling.trainer import trainer
from treemort.evaluation.evaluator import evaluator

from transformers import MaskFormerImageProcessor, AutoImageProcessor


def run(conf, eval_only):
    assert os.path.exists(
        conf.data_folder
    ), f"Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset, image_processor = prepare_datasets(conf.data_folder, conf=conf)

    model, optimizer, criterion, metrics = resume_or_load(conf, id2label, device)

    n_batches = len(train_dataset)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if eval_only:
        print("Evaluation-only mode started.")

        evaluator(
            model,
            dataset=test_dataset,
            num_samples=len(test_dataset),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            model_name=conf.model,
            image_processor=image_processor,
        )

    else:
        print("Training mode started.")

        trainer(
            model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            train_loader=train_dataset,
            val_loader=val_dataset,
            conf=conf,
            callbacks=callbacks,
            device=device,
            image_processor=image_processor,
        )


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
