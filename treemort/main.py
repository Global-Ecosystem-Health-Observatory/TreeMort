import os
import torch
import logging
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup


logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO, you can change to DEBUG if needed
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for logs
)
logger = logging.getLogger(__name__)  # Set logger name


def run(conf, eval_only):
    assert os.path.exists(conf.data_folder), f"[ERROR] Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        logger.info(f"Created output directory: {conf.output_dir}")
    else:
        logger.info(f"Output directory already exists: {conf.output_dir}")

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(conf)
    logger.info(f"Datasets prepared: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")

    logger.info("Loading or resuming model...")
    model, optimizer, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_dataset), device)
    logger.info("Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        logger.info("Evaluation-only mode started.")
        evaluator(
            model,
            dataset=test_dataset,
            num_samples=len(test_dataset),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            model_name=conf.model,
        )
        logger.info("Evaluation completed.")

    else:
        logger.info("Training mode started.")
        trainer(
            model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            train_loader=train_dataset,
            val_loader=val_dataset,
            conf=conf,
            callbacks=callbacks,
        )
        logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument(     "config", type=str,            help="Path to the configuration file")
    parser.add_argument("--eval-only", action="store_true", help="If set, only evaluate the model without training",)

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)


'''
Usage:

- Train

sh ~/TreeMort/scripts/run_treemort.sh ~/TreeMort/configs/flair_unet_bs8_cs256.txt --eval-only false

- Evaluate

sh ~/TreeMort/scripts/run_treemort.sh ~/TreeMort/configs/flair_unet_bs8_cs256.txt --eval-only true

'''