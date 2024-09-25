import torch
import logging

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.checkpoints import get_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building model...")

    model, optimizer, criterion, metrics = build_model(conf, id2label, device)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if conf.resume:
        load_checkpoint_if_available(model, conf)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, criterion, metrics, callbacks


def load_checkpoint_if_available(model, conf):
    checkpoint_path = get_checkpoint(conf.model_weights, conf.output_dir)

    if checkpoint_path:
        device = next(model.parameters()).device  # Get the device of the model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {checkpoint_path}.")
    else:
        logger.info("No checkpoint found. Training from scratch.")


def build_model(conf, id2label, device):
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer = configure_optimizer(model, conf.learning_rate)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, criterion, metrics
