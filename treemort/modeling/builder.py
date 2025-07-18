import torch

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint

logger = get_logger(__name__)

import itertools

# Define grids for each hyperparameter around defaults
mask_values = [0.8, 1.0, 1.2]  # Around default 1.0
centroid_values = [0.5, 0.7, 0.9]  # Around default 0.7
sdt_values = [0.3, 0.5, 0.7]  # Around default 0.5
boundary_values = [0.8, 1.0, 1.2]  # Around default 1.0

# Generate all combinations as list of dicts
combinations = [
    {
        'mask_weight': mask,
        'centroid_weight': centroid,
        'sdt_weight': sdt,
        'boundary_weight': boundary
    }
    for mask, centroid, sdt, boundary in itertools.product(
        mask_values, centroid_values, sdt_values, boundary_values
    )
]

combo = combinations[0]


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building model...")

    model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    if conf.model in ("flair_unet_baseline", "flair_unet_pretrained", "flair_unet_attention"):
        callbacks = build_callbacks(n_batches, conf.output_dir, optimizer, model_name=f"best.weights.{conf.model}.pth")
    else:
        callbacks = build_callbacks(n_batches, conf.output_dir, optimizer, model_name=f"best.weights.{conf.model}.{combo['mask_weight']}.{combo['centroid_weight']}.{combo['sdt_weight']}.{combo['boundary_weight']}.pth")

    if conf.resume:
        load_checkpoint_if_available(model, conf)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, schedular, criterion, metrics, callbacks


def load_checkpoint_if_available(model, conf):
    checkpoint_path = get_checkpoint(conf.model_weights, conf.output_dir)

    if checkpoint_path:
        device = next(model.parameters()).device  # Get the device of the model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Loaded weights from {checkpoint_path}.")
    else:
        logger.info("No checkpoint found. Training from scratch.")


def build_model(conf, id2label, device, total_steps=1):
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer, scheduler = configure_optimizer(model, conf.learning_rate, total_steps)

    if conf.model in ("flair_unet_baseline", "flair_unet_pretrained", "flair_unet_attention"):
        criterion, metrics = configure_loss_and_metrics(conf, use_multi_task=False)
    else:
        # criterion, metrics = configure_loss_and_metrics(conf)
        criterion, metrics = configure_loss_and_metrics(conf, sdt_weight=combo['sdt_weight'], boundary_weight=combo['boundary_weight'], mask_weight=combo['mask_weight'], centroid_weight=combo['centroid_weight'])

    return model, optimizer, scheduler, criterion, metrics