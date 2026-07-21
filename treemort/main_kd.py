import os
import copy
import torch
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.modeling.model_config import configure_model
from treemort.training.train_loop import (
    train_one_epoch_distillation,
    train_one_epoch_self_distillation,
    train_one_epoch_feature_level_distillation,
    train_one_epoch_ensemble_distillation,
)
from treemort.training.validation_loop import validate_one_epoch
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger

logger = get_logger(__name__)


def _load_teacher(conf, id2label, device, model_name=None, model_file=None):
    """Load a frozen teacher model from a checkpoint file."""
    teacher_conf = copy.copy(conf)
    if model_name is not None:
        teacher_conf.model = model_name

    if model_file:
        model_file = os.path.expandvars(model_file)

    teacher = configure_model(teacher_conf, id2label)
    teacher.to(device)
    teacher.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=True)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def run(conf, eval_only):
    assert os.path.exists(conf.data_folder), (
        f"[ERROR] Data folder {conf.data_folder} does not exist."
    )

    os.makedirs(conf.output_dir, exist_ok=True)
    run_dir = getattr(conf, "run_dir", os.path.join(conf.output_dir, conf.model))
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Set default distillation method if not provided
    if not hasattr(conf, 'distillation_method') or not conf.distillation_method:
        conf.distillation_method = 'basic'
    logger.info(f"Distillation method: {conf.distillation_method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    id2label = {0: "alive", 1: "dead"}

    if eval_only:
        conf.resume = True
        conf.best_model = f"best.weights.{conf.distillation_method}.pth"

    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets(conf)
    train_len = len(train_loader) if train_loader is not None else 0
    val_len = len(val_loader) if val_loader is not None else 0
    test_len = len(test_loader) if test_loader is not None else 0
    logger.info(f"Datasets prepared: Train({train_len}), Val({val_len}), Test({test_len})")

    # Load student model via resume_or_load
    logger.info("Loading student model...")
    num_steps = train_len if train_len > 0 else test_len
    student_model, optimizer, scheduler, criterion, metrics, callbacks = resume_or_load(
        conf, id2label, num_steps, device
    )
    logger.info("Student model loaded.")

    if eval_only:
        if test_loader is None or test_len == 0:
            raise RuntimeError("Evaluation requested but no test_loader is available.")
        logger.info("Evaluation-only mode started.")
        evaluator(student_model, test_loader, test_len, metrics, conf)
        logger.info("Evaluation completed.")
        return

    # Load teacher model(s)
    teacher_model_names = getattr(conf, 'teacher_model_names', conf.model)
    teacher_model_file_names = getattr(conf, 'teacher_model_file_names', None)

    if teacher_model_file_names is None:
        raise ValueError("--teacher-model-file-names must be set for KD training.")

    logger.info("Loading teacher model(s)...")
    if conf.distillation_method == 'ensemble':
        # Multiple teachers
        if isinstance(teacher_model_names, list):
            names = teacher_model_names
        else:
            names = [teacher_model_names]
        if isinstance(teacher_model_file_names, list):
            files = teacher_model_file_names
        else:
            files = [teacher_model_file_names]
        teacher_model = [
            _load_teacher(conf, id2label, device, model_name=n, model_file=f)
            for n, f in zip(names, files)
        ]
        logger.info(f"Loaded {len(teacher_model)} teacher model(s) for ensemble distillation.")
    else:
        # Single teacher
        t_name = teacher_model_names if isinstance(teacher_model_names, str) else teacher_model_names[0]
        t_file = teacher_model_file_names if isinstance(teacher_model_file_names, str) else teacher_model_file_names[0]

        if conf.distillation_method == 'self':
            # EMA model initialised as a copy of the student
            teacher_model = copy.deepcopy(student_model)
            teacher_model.to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            logger.info("EMA teacher initialised from student weights (self-distillation).")
        else:
            teacher_model = _load_teacher(conf, id2label, device, model_name=t_name, model_file=t_file)
            logger.info(f"Teacher loaded from: {t_file}")

    kd_criterion = torch.nn.KLDivLoss()

    alpha       = getattr(conf, 'distillation_alpha',       0.5)
    temperature = getattr(conf, 'distillation_temperature', 2.0)
    beta        = getattr(conf, 'distillation_beta',        0.999)
    lambda_feat = getattr(conf, 'distillation_lambda',      0.2)
    sharpen_t   = getattr(conf, 'distillation_sharpen_temperature', None)

    # Shared kwargs forwarded to every loss_fn
    kd_kwargs = dict(
        alpha=alpha,
        temperature=temperature,
    )
    if sharpen_t is not None:
        kd_kwargs['sharpen_temperature'] = sharpen_t

    teacher_name = (
        teacher_model_names
        if isinstance(teacher_model_names, str)
        else (teacher_model_names[0] if not isinstance(teacher_model_names, list) or conf.distillation_method != 'ensemble' else teacher_model_names)
    )

    best_val_iou = float('-inf')
    best_ckpt = os.path.join(run_dir, f"best.weights.{conf.distillation_method}.pth")

    for epoch in range(conf.epochs):
        logger.info(f"Epoch {epoch + 1}/{conf.epochs}")

        if conf.distillation_method == 'basic':
            train_loss, train_metrics = train_one_epoch_distillation(
                student_model=student_model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                kd_criterion=kd_criterion,
                metrics=metrics,
                train_loader=train_loader,
                model_name=conf.model,
                teacher_model_name=teacher_name,
                device=device,
                **kd_kwargs,
            )

        elif conf.distillation_method == 'self':
            train_loss, train_metrics = train_one_epoch_self_distillation(
                student_model=student_model,
                ema_model=teacher_model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                kd_criterion=kd_criterion,
                metrics=metrics,
                train_loader=train_loader,
                model_name=conf.model,
                teacher_model_name=conf.model,
                device=device,
                beta=beta,
                **kd_kwargs,
            )

        elif conf.distillation_method == 'feature':
            train_loss, train_metrics = train_one_epoch_feature_level_distillation(
                student_model=student_model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                kd_criterion=kd_criterion,
                metrics=metrics,
                train_loader=train_loader,
                model_name=conf.model,
                teacher_model_name=teacher_name,
                device=device,
                lambda_feature=lambda_feat,
                **kd_kwargs,
            )

        elif conf.distillation_method == 'ensemble':
            train_loss, train_metrics = train_one_epoch_ensemble_distillation(
                student_model=student_model,
                teacher_models=teacher_model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                kd_criterion=kd_criterion,
                metrics=metrics,
                train_loader=train_loader,
                model_name=conf.model,
                teacher_model_names=teacher_model_names,
                device=device,
                **kd_kwargs,
            )

        else:
            raise ValueError(f"Unknown distillation_method: {conf.distillation_method!r}")

        val_loss, val_metrics = validate_one_epoch(
            student_model, criterion, metrics, val_loader, conf, device
        )

        val_iou = val_metrics.get('iou_segments', float('-inf'))
        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(student_model.state_dict(), best_ckpt)
            logger.info(f"Saved best model checkpoint → {best_ckpt}  (val_iou={val_iou:.4f})")

    logger.info(f"KD training completed. Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation training entry point.")
    parser.add_argument("config",        type=str, help="Path to the model configuration file")
    parser.add_argument("--data-config", type=str, required=True, help="Path to the data configuration file")
    parser.add_argument('--verbosity',   type=str, default='info', choices=['info', 'debug', 'warning'])
    parser.add_argument("--eval-only",   action="store_true", help="If set, only evaluate without training")

    args = parser.parse_args()

    _ = configure_logger(verbosity=args.verbosity)

    conf = setup(args.config, data_config=args.data_config)

    run(conf, args.eval_only)


'''

1) Local

Usage: python3 -m treemort.main_kd <config file> --data-config <data config file> [--eval-only]

Example:

(Train) python3 -m treemort.main_kd $TREEMORT_REPO_PATH/configs/model/flair_unet_kd_basic.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt
(Test)  python3 -m treemort.main_kd $TREEMORT_REPO_PATH/configs/model/flair_unet_kd_basic.txt --data-config $TREEMORT_REPO_PATH/configs/data/finland.txt --eval-only

2) HPC

Usage: ./submit_treemort_kd.sh <hpc_type> <model config> <data config> [--eval-only]

Examples:

export TREEMORT_REPO_PATH="/users/aurahman/TreeMort"
export TREEMORT_TEACHER_PATH="/path/to/best.weights.pth"

(train) bash $TREEMORT_REPO_PATH/scripts/submit_treemort_kd.sh lumi flair_unet_kd_basic finland
(test)  bash $TREEMORT_REPO_PATH/scripts/submit_treemort_kd.sh lumi flair_unet_kd_basic finland --eval-only

'''
