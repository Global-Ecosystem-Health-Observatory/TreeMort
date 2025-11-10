import os
import torch
import torch.nn as nn

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint

logger = get_logger(__name__)


def resume_or_load(conf, id2label, n_batches, device):
    logger.info("Building model...")

    model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    callbacks = build_callbacks(n_batches, os.path.join(conf.output_dir, conf.model), optimizer)

    if conf.resume:
        load_checkpoint_if_available(model, conf)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, schedular, criterion, metrics, callbacks


def _slice_first_conv_weights(ckpt_state, model_state, rgb_indices=(1, 2, 3)):
    """
    Adapt a 4-channel first-conv (assumed order [NIR,R,G,B]) from the checkpoint
    to the model's 3-channel first-conv by selecting RGB slices.
    Finds a pair of conv weight tensors where ckpt has in_ch==4 and model has in_ch==3
    with the same out_ch and kernel size, then slices.
    Returns a possibly modified checkpoint state and a boolean indicating if surgery occurred.
    """
    # Gather candidate conv weight keys (format: <module>.weight)
    ckpt_conv_keys = [k for k, v in ckpt_state.items() if k.endswith('weight') and hasattr(v, 'shape') and len(v.shape) == 4]
    model_conv_keys = [k for k, v in model_state.items() if k.endswith('weight') and hasattr(v, 'shape') and len(v.shape) == 4]

    # Build quick index by (out_ch, kH, kW, in_ch)
    def sig(v):
        return (int(v.shape[0]), int(v.shape[2]), int(v.shape[3]), int(v.shape[1]))

    ckpt_candidates = {k: sig(ckpt_state[k]) for k in ckpt_conv_keys if ckpt_state[k].shape[1] == 4}
    model_candidates = {k: sig(model_state[k]) for k in model_conv_keys if model_state[k].shape[1] == 3}

    # Try to find a matching pair by out_ch and kernel size (ignore in_ch)
    for ck_k, (out_c, kH, kW, in_c_ck) in ckpt_candidates.items():
        for md_k, (out_c_md, kH_md, kW_md, in_c_md) in model_candidates.items():
            if out_c == out_c_md and kH == kH_md and kW == kW_md:
                # Perform slicing
                w4 = ckpt_state[ck_k]
                if w4.shape[1] != 4:
                    continue
                # Select RGB slices from [NIR,R,G,B] → indices [1,2,3]
                rgb_idx = torch.tensor(list(rgb_indices), dtype=torch.long, device=w4.device)
                w3 = w4.index_select(dim=1, index=rgb_idx)
                # Load into the model's expected key by overwriting checkpoint entry for that key
                ckpt_state[md_k] = w3
                # Remove the old 4ch key if names differ to avoid strict loading errors
                if md_k != ck_k and ck_k in ckpt_state:
                    del ckpt_state[ck_k]
                return ckpt_state, True

    return ckpt_state, False


def _freeze_encoder_blocks(model, keep_first_n=1):
    """
    Freeze encoder blocks 2..N (0-based indexing, keep_first_n un-frozen), with
    best-effort support for common U-Net encoders. Falls back to freezing most
    of the backbone while keeping the very first Conv2d trainable if exact
    structure is unknown.
    """
    # Try common attributes first
    try:
        enc = getattr(model, 'encoder', None)
        blocks = None
        if enc is not None:
            if hasattr(enc, 'blocks'):
                blocks = list(enc.blocks)
            elif hasattr(enc, 'stages'):
                blocks = list(enc.stages)
            elif hasattr(enc, 'layer1') and hasattr(enc, 'layer2'):
                # ResNet-like
                blocks = [b for b in [getattr(enc, 'layer1', None), getattr(enc, 'layer2', None), getattr(enc, 'layer3', None), getattr(enc, 'layer4', None)] if b is not None]
        if blocks:
            for i, blk in enumerate(blocks):
                if i < keep_first_n:
                    continue
                for p in blk.parameters():
                    p.requires_grad = False
            logger.info(f"Froze encoder blocks {keep_first_n}..{len(blocks)-1} (kept first {keep_first_n} trainable).")
            return True
    except Exception as e:
        logger.warning(f"Encoder-specific freezing failed with: {e}. Falling back to generic freeze.")

    # Fallback: freeze everything except parameters belonging to the very first Conv2d encountered
    first_conv_seen = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not first_conv_seen:
            first_conv_seen = True
            continue
        for p in getattr(module, 'parameters', lambda: [])():
            p.requires_grad = False
    logger.info("Applied fallback freezing: kept first Conv2d trainable, froze the rest.")
    return True


def load_checkpoint_if_available(model, conf):
    checkpoint_path = get_checkpoint(conf.model_weights, os.path.join(conf.output_dir, conf.model))

    if checkpoint_path:
        device = next(model.parameters()).device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Some checkpoints store under 'state_dict'
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        # Attempt 4→3 first-conv surgery if needed
        model_state = model.state_dict()
        try:
            state, did_surgery = _slice_first_conv_weights(state, model_state, rgb_indices=(1, 2, 3))
            if did_surgery:
                logger.info("Adapted first conv from 4→3 channels by selecting RGB slices (dropped NIR).")
        except Exception as e:
            logger.warning(f"First-conv channel surgery skipped due to error: {e}")

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
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer, scheduler = configure_optimizer(model, conf.learning_rate, total_steps)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, scheduler, criterion, metrics