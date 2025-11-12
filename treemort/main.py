import os
import torch
import argparse
from pathlib import Path

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger
from treemort.utils.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    use_dataset_artifact,
)

logger = get_logger(__name__)


def _maybe_use_dataset_artifact(conf, wandb_run):
    if not getattr(conf, "dataset_artifact", None):
        return {}

    artifact_download_root = Path(conf.output_dir) / conf.model / "dataset_artifacts"
    artifact_download_root.mkdir(parents=True, exist_ok=True)
    metadata = use_dataset_artifact(wandb_run, conf.dataset_artifact, str(artifact_download_root))

    if metadata.get("dataset_artifact_dir"):
        conf.data_folder = metadata["dataset_artifact_dir"]
    if metadata.get("dataset_hdf5_file"):
        conf.hdf5_file = metadata["dataset_hdf5_file"]

    return metadata


def run(conf, eval_only, wandb_run=None):
    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        logger.info(f"Created output directory: {conf.output_dir}")
    else:
        logger.info(f"Output directory already exists: {conf.output_dir}")

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_meta = {}
    if getattr(conf, "wandb", False) and getattr(conf, "dataset_artifact", None):
        dataset_meta = _maybe_use_dataset_artifact(conf, wandb_run)

    if not os.path.exists(conf.data_folder):
        raise FileNotFoundError(f"Data folder {conf.data_folder} does not exist even after artifact resolution.")

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
        conf, id2label, num_steps_for_setup, device, wandb_run=wandb_run
    )
    logger.info("Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        if test_loader is None or test_len == 0:
            raise RuntimeError("Evaluation requested but no test_loader is available (got None or empty).")
        logger.info("Evaluation-only mode started.")
        evaluator(model, test_loader, test_len, metrics, conf, wandb_run=wandb_run)
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
            wandb_run=wandb_run,
            dataset_meta=dataset_meta,
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
    config_paths = [args.config, args.data_config]
    wandb_run = init_wandb_run(conf, config_paths=config_paths)
    setattr(conf, "wandb_run", wandb_run)

    try:
        run(conf, args.eval_only, wandb_run=wandb_run)
    finally:
        finish_wandb_run(wandb_run)


'''

1) Local

Usage: python3 -m treemort.main <config file> --data-config <data config file> [--eval-only]

Example:

(Train) python3 -m treemort.main $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt
(Test)  python3 -m treemort.main $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt --eval-only

2) HPC

Usage: ./submit_treemort.sh <hpc_type> <model config file> <data config file> [--eval-only]

Examples:

export TREEMORT_REPO_PATH="/users/aurahman/TreeMort"

(train) bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland
(test)  bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland --eval-only

'''
