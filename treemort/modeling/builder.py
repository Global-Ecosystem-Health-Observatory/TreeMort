import os
import torch
import torch.nn as nn

from treemort.modeling.model_config import configure_model
from treemort.modeling.callback_builder import build_callbacks
from treemort.modeling.optimizer_loss_config import configure_optimizer, configure_loss_and_metrics

from treemort.utils.logger import get_logger
from treemort.utils.checkpoints import get_checkpoint


def resume_or_load(conf, id2label, n_batches, device):
    logger = get_logger()
    
    logger.info("Building model...")

    model, optimizer, schedular, criterion, metrics = build_model(conf, id2label, device, total_steps=conf.epochs * n_batches)

    run_dir = getattr(conf, 'run_dir', os.path.join(conf.output_dir, conf.model))
    callbacks = build_callbacks(
        n_batches,
        run_dir,
        optimizer,
        best_model=getattr(conf, 'best_model', 'best.weights.pth')
    )

    if conf.resume:
        load_checkpoint_if_available(model, conf, run_dir)
    else:
        logger.info("Training model from scratch.")

    return model, optimizer, schedular, criterion, metrics, callbacks


def _match_first_conv_channels(ckpt_state, model_state, rgb_indices=(1, 2, 3)):
    """
    Match checkpoint first-conv weights to target in/out channels by slicing or padding.
    Handles both 4->3 (drop NIR) and 3->4 (synthetic extra channel) cases.
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
                w = ckpt_state[ck_k]
                if w.shape[1] == in_c_md:
                    ckpt_state[md_k] = w
                elif w.shape[1] > in_c_md:
                    rgb_idx = torch.tensor(list(rgb_indices[:in_c_md]), dtype=torch.long, device=w.device)
                    ckpt_state[md_k] = w.index_select(dim=1, index=rgb_idx)
                else:
                    pad_ch = in_c_md - w.shape[1]
                    mean_channel = w.mean(dim=1, keepdim=True)
                    extra = mean_channel.repeat(1, pad_ch, 1, 1)
                    ckpt_state[md_k] = torch.cat([w, extra], dim=1)
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

        # Attempt 4→3 first-conv surgery if needed
        model_state = model.state_dict()
        try:
            state, did_surgery = _match_first_conv_channels(state, model_state, rgb_indices=(1, 2, 3))
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
    logger = get_logger()
    
    model = configure_model(conf, id2label)
    model.to(device)
    logger.info(f"Model successfully moved to {device}.")

    optimizer, scheduler = configure_optimizer(model, conf.learning_rate, total_steps)
    criterion, metrics = configure_loss_and_metrics(conf)

    return model, optimizer, scheduler, criterion, metrics
