import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint


def resume_or_load(conf, id2label, n_batches, device, is_main=True):
    logger = get_logger()

    logger.info("Building model...")

    model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    run_dir = getattr(conf, 'run_dir', os.path.join(conf.output_dir, conf.model))
    callbacks = build_callbacks(
        n_batches,
        run_dir,
        optimizer,
        best_model=getattr(conf, 'best_model', 'best.weights.pth'),
        is_main=is_main,
    )

    if conf.resume:
        load_checkpoint_if_available(model, conf, run_dir)
    else:
        logger.info("Training model from scratch.")

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[0])

    return model, optimizer, schedular, criterion, metrics, callbacks


def _freeze_encoder_blocks(model, keep_first_n=1):
    """
    Freeze encoder blocks 2..N (0-based indexing, keep_first_n un-frozen), with
    best-effort support for common U-Net encoders. Falls back to freezing most
    of the backbone while keeping the very first Conv2d trainable if exact
    structure is unknown.
    """
    logger = get_logger()

    enc = getattr(model, 'encoder', None)

    if enc is None and hasattr(model, 'feature_extractor'):
        fe = getattr(model, 'feature_extractor')
        base = getattr(fe, 'model', None)
        if base is not None:
            seg_model = getattr(base, 'seg_model', None)
            if seg_model is not None:
                enc = getattr(seg_model, 'encoder', None)

    if enc is None:
        logger.warning("Freeze requested but encoder structure not found; skipping encoder freezing.")
        return False

    blocks = None
    if hasattr(enc, 'blocks'):
        blocks = list(enc.blocks)
    elif hasattr(enc, 'stages'):
        blocks = list(enc.stages)
    elif hasattr(enc, 'layer1') and hasattr(enc, 'layer2'):
        blocks = [
            b
            for b in [
                getattr(enc, 'layer1', None),
                getattr(enc, 'layer2', None),
                getattr(enc, 'layer3', None),
                getattr(enc, 'layer4', None)
            ]
            if b is not None
        ]

    if not blocks:
        logger.warning("Encoder blocks not detected; skipping freeze.")
        return False

    for i, blk in enumerate(blocks):
        if i < keep_first_n:
            continue
        for p in blk.parameters():
            p.requires_grad = False

    logger.info(f"Froze encoder blocks {keep_first_n}..{len(blocks)-1} (kept first {keep_first_n} trainable).")
    return True


def load_checkpoint_if_available(model, conf, run_dir):
    logger = get_logger()

    checkpoint_path = getattr(conf, 'resume_from', None)
    if checkpoint_path:
        checkpoint_path = os.path.expandvars(checkpoint_path)
    if not checkpoint_path:
        checkpoint_path = get_checkpoint(
            conf.model_weights,
            run_dir,
            getattr(conf, 'best_model', 'best.weights.pth')
        )

    if checkpoint_path:
        device = next(model.parameters()).device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Some checkpoints store under 'state_dict'
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        # Load with strict=False to allow minor key mismatches (e.g., num_batches_tracked)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.info(f"Missing keys when loading: {missing}")
        if unexpected:
            logger.info(f"Unexpected keys when loading: {unexpected}")
        logger.info(f"Loaded weights from {checkpoint_path}.")

        # Optional: initial freezing of encoder blocks (progressive unfreeze handled by callbacks/train loop)
        freeze_epochs = getattr(conf, 'freeze_epochs', 0)
        if freeze_epochs and freeze_epochs > 0:
            kept = getattr(conf, 'keep_first_encoder_blocks', 1)
            _freeze_encoder_blocks(model, keep_first_n=int(kept))
            # Stash hint for the training loop/callbacks to unfreeze later
            setattr(model, '_unfreeze_after_epochs', int(freeze_epochs))
            logger.info(f"Encoder blocks frozen for first {freeze_epochs} epoch(s); will unfreeze progressively thereafter.")
    else:
        logger.info("No checkpoint found. Training from scratch.")


def build_model(conf, id2label, device, total_steps=1):
    logger = get_logger()

    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer, scheduler = configure_optimizer(model, conf.learning_rate, total_steps)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, scheduler, criterion, metrics
