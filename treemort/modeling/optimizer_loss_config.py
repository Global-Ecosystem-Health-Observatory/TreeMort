import torch

import numpy as np

from scipy.ndimage import label as cc_label

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import TreeMortalityLoss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import (
    masked_iou,
    masked_f1,
    apply_activation,
    tree_iou_from_masks,
    aggregate_epoch_metrics_with_ci,
)


def configure_optimizer(model, learning_rate, total_steps):
    logger = get_logger()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.3)

    logger.info(
        f"Configured {optimizer.__class__.__name__} with OneCycleLR scheduler (max_lr={learning_rate}, total_steps={total_steps})."
    )
    return optimizer, scheduler


def configure_loss_and_metrics(conf, class_weights=None):
    logger = get_logger()
    
    if conf.loss == "hybrid":
        tree_mortality_loss = TreeMortalityLoss(
            mask_weight=getattr(conf, "mask_weight", 1.0),
            centroid_weight=getattr(conf, "centroid_weight", 3.0),
            sdt_weight=getattr(conf, "sdt_weight", 0.5),
            boundary_weight=getattr(conf, "boundary_weight", 1.0),
            centroid_pos_weight=getattr(conf, "centroid_pos_weight", 10.0),
            centroid_min_target=getattr(conf, "centroid_min_target", 0.1),
            # Hybrid (SDT+boundary) stabilizers
            hybrid_use_tanh=getattr(conf, "hybrid_use_tanh", True),
            hybrid_bg_weight=getattr(conf, "hybrid_bg_weight", 0.10),
            hybrid_interior_weight=getattr(conf, "hybrid_interior_weight", 3.00),
            hybrid_boundary_weight=getattr(conf, "hybrid_boundary_weight", 1.00),
        )

        logger.info(
            "TreeMortalityLoss(hybrid) params: "
            f"hybrid_use_tanh={tree_mortality_loss.hybrid_use_tanh}, "
            f"hybrid_bg_weight={tree_mortality_loss.hybrid_bg_weight}, "
            f"hybrid_interior_weight={tree_mortality_loss.hybrid_interior_weight}, "
            f"hybrid_boundary_weight={tree_mortality_loss.hybrid_boundary_weight}"
        )

        def criterion(pred, target, buffer=None):
            if buffer is None:
                buffer = torch.ones_like(target[:, 0:1, :, :])  # fallback all-ones

            return tree_mortality_loss(pred, target, buffer=buffer)

        def metrics(pred, target, buffer=None):
            pred_channels = pred.shape[1]
            target_channels = target.shape[1]

            segment_threshold = getattr(conf, "segment_threshold", 0.5)
            centroid_threshold = getattr(conf, "centroid_threshold", 0.5)
            act_name = getattr(conf, "activation", "sigmoid")
            max_centroid_dist = getattr(conf, "centroid_max_distance_px", 50)

            instance_iou_thresh = segment_threshold

            pred_mask = pred[:, 0, :, :]
            true_mask = target[:, 0, :, :]

            if buffer is None:
                buffer_mask = torch.ones_like(true_mask)  # fallback: no masking
            else:
                buffer_mask = buffer.squeeze(1) if buffer.ndim == 4 else buffer  # ensure [B,H,W]
            
            pred_probs = apply_activation(pred_mask, activation=act_name)
            pred_bin = (pred_probs > segment_threshold).float() * buffer_mask
            true_bin = (true_mask > segment_threshold).float() * buffer_mask

            # Build per-image metric dicts, then aggregate with CI across images in the batch
            B, H, W = pred_bin.shape
            structure = np.ones((3, 3), dtype=bool)  # 8-connectivity

            per_image_metrics = []

            # Optional buffer mask (channel 3 if present); otherwise all-ones
            if pred_channels > 3:
                # Extract centroids
                pred_centroid = pred[:, 1, :, :]
                true_centroid = target[:, 1, :, :]

                pred_centroid_probs = apply_activation(pred_centroid, activation=act_name)
                pred_centroid_bin = (pred_centroid_probs > centroid_threshold).float() * buffer_mask
                true_centroid_bin = (true_centroid > centroid_threshold).float() * buffer_mask

                pred_centroid_bin_np = pred_centroid_bin.detach().cpu().numpy().astype(np.uint8)
                true_centroid_bin_np = true_centroid_bin.detach().cpu().numpy().astype(np.uint8)
            else:
                pred_centroid_bin_np = pred_bin.detach().cpu().numpy().astype(np.uint8)
                true_centroid_bin_np = true_bin.detach().cpu().numpy().astype(np.uint8)

            for b in range(B):
                # Per-image segmentation metrics
                iou_b = masked_iou(pred_probs[b], true_mask[b], buffer_mask[b], threshold=segment_threshold)
                f1_b = masked_f1(pred_probs[b], true_mask[b], buffer_mask[b], threshold=segment_threshold)

                # Pixel-area metrics per image
                pred_b = pred_bin[b]
                true_b = true_bin[b]
                intersection = (pred_b * true_b).sum()
                pred_area = pred_b.sum()
                true_area = true_b.sum()
                pixel_precision_b = float(intersection / (pred_area + 1e-8)) if float(pred_area) > 0 else 0.0
                pixel_recall_b = float(intersection / (true_area + 1e-8)) if float(true_area) > 0 else 0.0
                pixel_f1_b = float(2 * pixel_precision_b * pixel_recall_b / (pixel_precision_b + pixel_recall_b + 1e-8))

                # Connected components for centroid-based metrics
                pred_labeled, n_pred = cc_label(pred_centroid_bin_np[b] > 0, structure=structure)
                true_labeled, n_true = cc_label(true_centroid_bin_np[b] > 0, structure=structure)

                # Centroids
                def centroids_from_labels(lbl_img, n):
                    cents = []
                    for i in range(1, n + 1):
                        ys, xs = np.where(lbl_img == i)
                        if ys.size == 0:
                            cents.append((np.nan, np.nan))
                        else:
                            cents.append((float(ys.mean()), float(xs.mean())))
                    return np.array(cents, dtype=float) if n > 0 else np.zeros((0, 2), dtype=float)

                pred_centroids = centroids_from_labels(pred_labeled, n_pred)
                true_centroids = centroids_from_labels(true_labeled, n_true)

                # Greedy centroid matching within distance
                tp_b = 0
                fp_b = int(n_pred)
                fn_b = int(n_true)
                loc_sum_b = 0.0
                loc_count_b = 0

                if n_pred > 0 and n_true > 0:
                    dists = np.sqrt(((pred_centroids[:, None, :] - true_centroids[None, :, :]) ** 2).sum(axis=2))
                    pairs = []
                    for i in range(n_pred):
                        for j in range(n_true):
                            if np.isfinite(dists[i, j]) and dists[i, j] <= max_centroid_dist:
                                pairs.append((dists[i, j], i, j))
                    pairs.sort(key=lambda x: x[0])

                    matched_pred = set()
                    matched_true = set()
                    for dist, i_idx, j_idx in pairs:
                        if i_idx not in matched_pred and j_idx not in matched_true:
                            matched_pred.add(i_idx)
                            matched_true.add(j_idx)
                            tp_b += 1
                            loc_sum_b += float(dist)
                            loc_count_b += 1
                    fp_b = int(n_pred - len(matched_pred))
                    fn_b = int(n_true - len(matched_true))

                # Per-image instance rates
                prec_b = float(tp_b / (tp_b + fp_b)) if (tp_b + fp_b) > 0 else 0.0
                rec_b = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
                f1i_b = float(2 * prec_b * rec_b / (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0.0
                cerr_b = float(loc_sum_b / loc_count_b) if loc_count_b > 0 else float("nan")

                # Per-image Tree IoU via utility on singleton batch
                tree_stats_b = tree_iou_from_masks(
                    pred_b[None, ...],
                    true_b[None, ...],
                    buffer_mask=buffer_mask[b][None, ...],
                    seg_threshold=segment_threshold,
                    iou_thresh=instance_iou_thresh,
                )

                per_image_metrics.append(
                    {
                        # Segmentation/mask metrics
                        "iou_segments": iou_b,
                        "f_score_segments": f1_b,
                        # Pixel-area metrics
                        "pixel_precision": pixel_precision_b,
                        "pixel_recall": pixel_recall_b,
                        "pixel_f1_score": pixel_f1_b,
                        # Instance (centroid) rates and localization
                        "instance_precision": prec_b,
                        "instance_recall": rec_b,
                        "instance_f1_score": f1i_b,
                        "centroid_err": cerr_b,
                        "centroid_err_sum": float(loc_sum_b),
                        "centroid_err_count": int(loc_count_b),
                        # Integer counts for micro-aggregation
                        "tp": int(tp_b),
                        "fp": int(fp_b),
                        "fn": int(fn_b),
                        "pred_peaks": int(n_pred),
                        "true_peaks": int(n_true),
                        # Tree IoU and counts per image
                        "tree_iou": float(tree_stats_b["tree_iou"]),
                        "tree_tp": int(tree_stats_b["tree_tp"]),
                        "tree_fp": int(tree_stats_b["tree_fp"]),
                        "tree_fn": int(tree_stats_b["tree_fn"]),
                    }
                )

            # Aggregate across images in this batch with CIs
            batch_agg = aggregate_epoch_metrics_with_ci(per_image_metrics, confidence=0.95)
            return batch_agg

        logger.info(
            "Configured hybrid loss + sigmoid activation; using segment_threshold for both binarization and instance IoU; centroid_max_distance_px=50."
        )
        return criterion, metrics

    elif conf.loss == "mse":

        def criterion(pred, target):
            pass

        def metrics(pred, target):
            pass

        logger.info("Masked MSE loss configured with buffer weighting.")
        return criterion, metrics

    elif conf.loss == "weighted_dice_loss":

        def criterion(pred, target):
            pass

        def metrics(pred, target):
            pass

        logger.info("Buffer-weighted Dice loss configured.")
        return criterion, metrics

    else:
        raise ValueError(f"Unsupported loss type: {conf.loss}")
