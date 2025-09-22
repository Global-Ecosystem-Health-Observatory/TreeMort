import os
import torch
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger

logger = get_logger(__name__)


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
    train_loader, val_loader, test_loader = prepare_datasets(conf)

    # Safely compute lengths when some loaders are None (e.g., test-only mode)
    train_len = len(train_loader) if train_loader is not None else 0
    val_len = len(val_loader) if val_loader is not None else 0
    test_len = len(test_loader) if test_loader is not None else 0

    logger.info(
        f"Datasets prepared: Train({train_len}), Val({val_len}), Test({test_len})"
    )

    # If test-only is enabled in config, force eval-only behavior
    if getattr(conf, "test_only", False) and not eval_only:
        logger.warning("`test_only` is True but `--eval-only` not set. Forcing evaluation-only mode.")
        eval_only = True

    logger.info("Loading or resuming model...")
    # Use a sensible length for model setup even in test-only mode
    num_steps_for_setup = train_len if train_len > 0 else test_len
    model, optimizer, schedular, criterion, metrics, callbacks = resume_or_load(
        conf, id2label, num_steps_for_setup, device
    )
    logger.info("Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        if test_loader is None or test_len == 0:
            raise RuntimeError("Evaluation requested but no test_loader is available (got None or empty).")
        logger.info("Evaluation-only mode started.")
        evaluator(model, test_loader, test_len, metrics, conf)
        logger.info("Evaluation completed.")

    else:
        if train_loader is None or val_loader is None:
            raise RuntimeError(
                "Training requested but train/val loaders are None. Set `--eval-only` or disable `test_only` in config."
            )
        logger.info("Training mode started.")
        trainer(
            model,
            optimizer=optimizer,
            schedular=schedular,
            criterion=criterion,
            metrics=metrics,
            train_loader=train_loader,
            val_loader=val_loader,
            conf=conf,
            callbacks=callbacks,
        )
        logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config",        type=str, help="Path to the configuration file")
    parser.add_argument("--data-config", type=str, required=True, help="Path to the additional data configuration file")
    parser.add_argument('--verbosity',   type=str, default='info', choices=['info', 'debug', 'warning'])
    parser.add_argument("--eval-only",   action="store_true", help="If set, only evaluate the model without training")
    
    args = parser.parse_args()
    
    _ = configure_logger(verbosity=args.verbosity)
    
    conf = setup(args.config, data_config=args.data_config)

    run(conf, args.eval_only)


'''

1) Local

Usage: python3 -m treemort.main <config file> --data-config <data config file> [--eval-only]

Example:

(Train) python3 -m treemort.main $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt
(Test)  python3 -m treemort.main $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt --eval-only

2) HPC

Usage: ./submit_treemort.sh <hpc_type> <model config file> <data config file> [--eval-only]

Examples:

export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

(train) bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland
(test)  bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland --eval-only

'''