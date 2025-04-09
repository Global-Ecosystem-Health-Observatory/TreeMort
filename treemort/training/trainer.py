import torch

from torch.utils.data import DataLoader
from typing import Callable, Any, List, Union

from treemort.training.train_loop import (
    train_one_epoch_distillation,
    train_one_epoch_self_distillation,
    train_one_epoch_feature_level_distillation,
    train_one_epoch_ensemble_distillation,
)
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.callbacks import EarlyStopping

logger = get_logger(__name__)


def trainer(
    student_model: torch.nn.Module,
    teacher_model_or_ema: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    kd_criterion: Callable,
    metrics: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conf: Any,
    callbacks: List[Any],
) -> None:

    device = next(student_model.parameters()).device

    if conf.distillation_method == "basic":
        training_loop = train_one_epoch_distillation
    elif conf.distillation_method == "self":
        training_loop = train_one_epoch_self_distillation
    elif conf.distillation_method == "feature":
        training_loop = train_one_epoch_feature_level_distillation
    elif conf.distillation_method == "ensemble":
        training_loop = train_one_epoch_ensemble_distillation
    else:
        raise ValueError("Unknown distillation method specified.")

    if isinstance(teacher_model_or_ema, list):
        for m in teacher_model_or_ema:
            m.eval()
    else:
        teacher_model_or_ema.eval()

    for epoch in range(conf.epochs):
        logger.info(f"Epoch {epoch + 1}/{conf.epochs} - Training ({conf.distillation_method} KD) started.")

        extra_params = {}
        if conf.distillation_method == "self":
            extra_params["beta"] = conf.distillation_beta
        elif conf.distillation_method == "feature":
            extra_params["lambda_feature"] = conf.distillation_lambda

        train_loss, train_metrics = training_loop(
            student_model,
            teacher_model_or_ema,
            optimizer,
            criterion,
            kd_criterion,
            metrics,
            train_loader,
            conf,
            device,
            alpha=conf.distillation_alpha,
            temperature=conf.distillation_temperature,
            **extra_params
        )

        logger.info(f"Epoch {epoch + 1} - Training completed.")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Metrics: {train_metrics}")

        val_loss, val_metrics = validate_one_epoch(
            student_model, criterion, metrics, val_loader, conf, device
        )

        logger.info(f"Epoch {epoch + 1} - Validation completed.")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Metrics: {val_metrics}")

        handle_callbacks(callbacks, epoch, student_model, optimizer, val_loss)

        if any(
            [isinstance(cb, EarlyStopping) and cb.stop_training for cb in callbacks]
        ):
            logger.info("Early stopping triggered.")
            break

    logger.info("Training process completed.")
