import os

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.wandb_utils import init_weights_from_registry
from treemort.utils.callbacks import EncoderFreezeCallback

logger = get_logger(__name__)


def resume_or_load(conf, id2label, n_batches, device, wandb_run=None):
    logger.info("Building model...")

    model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    callbacks = build_callbacks(n_batches, os.path.join(conf.output_dir, conf.model), optimizer)

    if getattr(conf, "freeze_epochs", 0) > 0 and getattr(conf, "keep_first_encoder_blocks", 0) > 0:
        callbacks.append(
            EncoderFreezeCallback(
                model,
                freeze_epochs=conf.freeze_epochs,
                keep_blocks=conf.keep_first_encoder_blocks,
            )
        )

    if getattr(conf, "init_model_artifact", None):
        init_weights_from_registry(model, conf.init_model_artifact, device, run=wandb_run)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, schedular, criterion, metrics, callbacks


def build_model(conf, id2label, device, total_steps=1):
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer, scheduler = configure_optimizer(model, conf.learning_rate, total_steps)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, scheduler, criterion, metrics
