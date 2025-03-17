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
    logger.info(f"Datasets prepared: Train({len(train_loader)}), Val({len(val_loader)}), Test({len(test_loader)})")

    logger.info("Loading or resuming model...")
    model, optimizer, schedular, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_loader), device)
    logger.info("Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        logger.info("Evaluation-only mode started.")
        evaluator(model, test_loader, len(test_loader), conf)
        logger.info("Evaluation completed.")

    else:
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
    
    conf = setup(args.config, args.data_config)

    run(conf, args.eval_only)


'''
Usage:

1) Train

python3 -m treemort.main ./configs/model/flair_unet.txt --data-config ./configs/data/finland.txt

2) Evaluate

python3 -m treemort.main ./configs/model/flair_unet.txt --data-config ./configs/data/finland.txt --eval-only

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

1) Train

sh $TREEMORT_REPO_PATH/scripts/run_treemort.sh $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt --eval-only false

2) Evaluate

sh $TREEMORT_REPO_PATH/scripts/run_treemort.sh $TREEMORT_REPO_PATH/configs/model/flair_unet.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt --eval-only true

'''