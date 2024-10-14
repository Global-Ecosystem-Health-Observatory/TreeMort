import os
import torch
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def run(conf, eval_only):
    #assert os.path.exists(conf.hdf5_file_finnish), f"[ERROR] Finnish data file {conf.hdf5_file_finnish} does not exist."
    #assert os.path.exists(conf.hdf5_file_us), f"[ERROR] US data file {conf.hdf5_file_us} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        logger.info(f"Created output directory: {conf.output_dir}")
    else:
        logger.info(f"Output directory already exists: {conf.output_dir}")

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader_finnish, test_loader_us = prepare_datasets(conf)
    logger.info(f"Datasets prepared: Train({len(train_loader)}), Val({len(val_loader)}), Test_Finnish({len(test_loader_finnish)}), Test_US({len(test_loader_us)})")

    logger.info("Loading or resuming model...")
    model, optimizer, seg_criterion, domain_criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_loader), device)
    logger.info("Model, optimizer, segmentation loss, domain loss, and metrics are set up.")

    if eval_only:
        logger.info("Evaluation-only mode started.")

        logger.info("Evaluating on Finnish test data...")
        evaluator(
            model,
            dataset=test_loader_finnish,
            num_samples=len(test_loader_finnish),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            model_name=conf.model,
        )
        logger.info("Finnish test data evaluation completed.")

        logger.info("Evaluating on US test data...")
        evaluator(
            model,
            dataset=test_loader_us,
            num_samples=len(test_loader_us),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            model_name=conf.model,
        )
        logger.info("US test data evaluation completed.")

    else:
        logger.info("Training mode started.")
        trainer(
            model=model,
            optimizer=optimizer,
            seg_criterion=seg_criterion,
            domain_criterion=domain_criterion,
            metrics=metrics,
            train_loader=train_loader,
            val_loader=val_loader,
            conf=conf,
            callbacks=callbacks,
        )
        logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--eval-only", action="store_true", help="If set, only evaluate the model without training")

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)