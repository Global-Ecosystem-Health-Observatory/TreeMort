import torch

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint

logger = get_logger(__name__)


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building domain-adversarial model...")
    
    model, optimizer, seg_criterion, metrics = build_model(conf, id2label, device)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if conf.resume:
        load_checkpoint_if_available(model, conf)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, seg_criterion, metrics, callbacks


def load_checkpoint_if_available(model, conf):
    checkpoint_path = get_checkpoint(conf.model_weights, conf.output_dir)

    if checkpoint_path:
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {checkpoint_path}.")
    else:
        logger.info("No checkpoint found. Training from scratch.")


def build_model(conf, id2label, device):
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer = configure_optimizer(model, conf.learning_rate)

    seg_criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, seg_criterion, metrics