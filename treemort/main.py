import os
import torch
import torch.distributed as dist
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger

logger = get_logger(__name__)


def run(conf, eval_only):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = rank == 0

    # ROCR_VISIBLE_DEVICES restricts each process to a single GPU exposed as device 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
        torch.cuda.set_device(0)
    if is_main:
        logger.info(f"Using device: {device}  |  world_size={world_size}")

    assert os.path.exists(conf.data_folder), f"[ERROR] Data folder {conf.data_folder} does not exist."

    run_dir = getattr(conf, "run_dir", os.path.join(conf.output_dir, conf.model))

    if is_main:
        os.makedirs(conf.output_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    if is_distributed:
        dist.barrier()

    id2label = {0: "alive", 1: "dead"}

    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets(conf, rank=rank, world_size=world_size)

    train_len = len(train_loader) if train_loader is not None else 0
    val_len = len(val_loader) if val_loader is not None else 0
    test_len = len(test_loader) if test_loader is not None else 0

    if is_main:
        logger.info(f"Datasets prepared: Train({train_len}), Val({val_len}), Test({test_len})")

    if getattr(conf, "test_only", False) and not eval_only:
        if is_main:
            logger.warning("`test_only` is True but `--eval-only` not set. Forcing evaluation-only mode.")
        eval_only = True

    if eval_only and not getattr(conf, "resume", False):
        if is_main:
            logger.info("Evaluation requested; forcing resume to load saved weights.")
        conf.resume = True

    if is_main:
        logger.info("Loading or resuming model...")
    num_steps_for_setup = train_len if train_len > 0 else test_len
    model, optimizer, schedular, criterion, metrics, callbacks = resume_or_load(
        conf, id2label, num_steps_for_setup, device, is_main=is_main
    )
    if is_main:
        logger.info("Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        if test_loader is None or test_len == 0:
            raise RuntimeError("Evaluation requested but no test_loader is available (got None or empty).")
        if is_main:
            logger.info("Evaluation-only mode started.")
        evaluator(model, test_loader, test_len, metrics, conf)
        if is_main:
            logger.info("Evaluation completed.")

    else:
        if train_loader is None or val_loader is None:
            raise RuntimeError(
                "Training requested but train/val loaders are None. Set `--eval-only` or disable `test_only` in config."
            )
        if is_main:
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
            is_main=is_main,
            is_distributed=is_distributed,
        )
        if is_main:
            logger.info("Training completed.")

    if is_distributed:
        dist.destroy_process_group()


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

export TREEMORT_REPO_PATH="/users/anisrahm/TreeMort"

(train) bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland
(test)  bash $TREEMORT_REPO_PATH/scripts/submit_treemort.sh lumi unet finland --eval-only

'''
