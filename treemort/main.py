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

    if eval_only:
        conf.resume = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets(conf)
    logger.info(f"Datasets prepared: Train({len(train_loader)}), Val({len(val_loader)}), Test({len(test_loader)})")
    # Set default distillation method if not provided in configuration
    if not hasattr(conf, 'distillation_method'):
        conf.distillation_method = 'basic'
    logger.info(f"Using distillation method: {conf.distillation_method}")

    logger.info("Loading or resuming models (student and teacher)...")
    student_model, teacher_model, optimizer, scheduler, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_loader), device)
    logger.info("Student and teacher models, optimizer, criterion, metrics, and callbacks are set up.")
    # Wrap teacher model in a list if ensemble distillation is used and teacher_model is not already a list
    if conf.distillation_method == 'ensemble' and not isinstance(teacher_model, list):
        teacher_model = [teacher_model]

    if eval_only:
        logger.info("Evaluation-only mode started.")
        evaluator(student_model, test_loader, len(test_loader), metrics, conf)
        logger.info("Evaluation completed.")

    else:
        logger.info("Training mode started.")

        kd_criterion = torch.nn.KLDivLoss()

        trainer(
            student_model=student_model,
            teacher_model_or_ema=teacher_model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            kd_criterion=kd_criterion,
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