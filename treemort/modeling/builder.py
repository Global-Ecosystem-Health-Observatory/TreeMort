import torch

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.checkpoints import get_checkpoint


def resume_or_load(conf, id2label, n_batches, device):
    print("[INFO] Building model...")

    model, optimizer, criterion, metrics = build_model(conf, id2label, device)

    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)

    if conf.resume:
        load_checkpoint_if_available(model, conf)
    else:
        print("[INFO] Training model from scratch.")

    return model, optimizer, criterion, metrics, callbacks


def load_checkpoint_if_available(model, conf):
    checkpoint_path = get_checkpoint(conf.model_weights, conf.output_dir)

    if checkpoint_path:
        device = next(model.parameters()).device  # Get the device of the model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] Loaded weights from {checkpoint_path}.")
    else:
        print("[INFO] No checkpoint found. Training from scratch.")


def build_model(conf, id2label, device):
    model = configure_model(conf, id2label)
    model.to(device)
    print(f"[INFO] Model successfully moved to {device}.")

    optimizer = configure_optimizer(model, conf.learning_rate)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, criterion, metrics
