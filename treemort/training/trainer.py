import torch

from tqdm import tqdm
from typing import Callable, Any, List, Union

from torch.utils.data import DataLoader

from treemort.training.train_loop import (
    train_one_epoch_distillation,
    train_one_epoch_self_distillation,
    train_one_epoch_feature_level_distillation,
    train_one_epoch_ensemble_distillation,
)

from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.metrics import log_metrics


def trainer(
    student_model: torch.nn.Module,
    teacher_model_or_ema: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: Callable,
    kd_criterion: Callable,
    metrics: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conf: Any,
    callbacks: List[Any],
) -> None:

    logger = get_logger()

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

    device = next(student_model.parameters()).device
    best_metric = float('inf')

    for epoch in tqdm(range(conf.epochs), desc="Epochs", unit="epoch"):
        logger.info(f"Epoch {epoch + 1}/{conf.epochs} - Training ({conf.distillation_method} KD) started.")

        extra_params = {}
        if conf.distillation_method == "self":
            extra_params["beta"] = conf.distillation_beta
        elif conf.distillation_method == "feature":
            extra_params["lambda_feature"] = conf.distillation_lambda
            extra_params["use_feature_loss"] = getattr(conf, "use_feature_loss", True)
            extra_params["use_confidence_weighting"] = getattr(conf, "use_confidence_weighting", True)
            extra_params["use_fg_bg_weighting"] = getattr(conf, "use_fg_bg_weighting", True)
            extra_params["sharpen_temperature"] = getattr(conf, "distillation_sharpen_temperature", 4.0)

        if hasattr(conf, "distillation_sharpen_temperature"):
            extra_params["sharpen_temperature"] = conf.distillation_sharpen_temperature

        initial_alpha = 0.5
        final_alpha = 0.2
        total_epochs = conf.epochs
        alpha = initial_alpha + (final_alpha - initial_alpha) * epoch / total_epochs

        initial_temp = 10.0
        final_temp = 4.0
        temperature = initial_temp + (final_temp - initial_temp) * epoch / total_epochs

        train_loss, train_metrics = training_loop(
            student_model,
            teacher_model_or_ema,
            optimizer,
            scheduler,
            criterion,
            kd_criterion,
            metrics,
            train_loader,
            conf.model,
            conf.teacher_model_names,
            device,
            alpha=alpha,
            temperature=temperature,
            **extra_params
        )

        val_loss, val_metrics = validate_one_epoch(
            student_model, criterion, metrics, 
            val_loader, conf.model, device
        )

        stop_training = handle_callbacks(
            callbacks,
            epoch=epoch,
            model=student_model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_metrics=val_metrics
        )

        if stop_training:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training completed successfully.")
    return student_model