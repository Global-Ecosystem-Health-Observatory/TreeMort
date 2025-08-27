import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import weighted_dice_loss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation, proximity_metrics, tree_iou_from_masks, aggregate_epoch_metrics_with_ci
import numpy as np
from scipy.ndimage import label as cc_label

logger = get_logger(__name__)


def configure_optimizer(model, learning_rate, total_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3
    )
    
    logger.info(f"Configured {optimizer.__class__.__name__} with OneCycleLR scheduler (max_lr={learning_rate}, total_steps={total_steps}).")
    return optimizer, scheduler


def configure_loss_and_metrics(conf, class_weights=None):
    if conf.loss == "hybrid":
        def criterion(pred, target):
            pred_mask = pred[:, 0, :, :]

            buffer_mask = target[:, 3, :, :]
            true_mask = target[:, 0, :, :]

            return hybrid_loss(pred_mask, true_mask, buffer_mask)

        def metrics(pred, target):
            # We assume a single-channel model output: segmentation mask
            pred_channels = pred.shape[1]
            target_channels = target.shape[1]

            # Thresholds and config
            seg_thresh = getattr(conf, "segment_threshold", 0.5)
            act_name = getattr(conf, "activation", "sigmoid")  # use sigmoid
            # Instance/centroid thresholds (in pixels)
            max_centroid_dist = getattr(conf, "centroid_max_distance_px", 50)
            instance_iou_thresh = getattr(conf, "instance_iou_threshold", 0.4)

            # Extract masks
            pred_mask = pred[:, 0, :, :]
            true_mask = target[:, 0, :, :]

            # Optional buffer mask (channel 3 if present); otherwise all-ones
            if target_channels > 3:
                buffer_mask = target[:, 3, :, :]
            else:
                buffer_mask = torch.ones_like(true_mask)

            # Probabilities and binarization
            pred_probs = apply_activation(pred_mask, activation=act_name)
            pred_bin = (pred_probs > seg_thresh).float() * buffer_mask
            true_bin = (true_mask > seg_thresh).float() * buffer_mask

            # Build per-image metric dicts, then aggregate with CI across images in the batch
            B, H, W = pred_bin.shape
            structure = np.ones((3, 3), dtype=bool)  # 8-connectivity

            per_image_metrics = []

            pred_bin_np = pred_bin.detach().cpu().numpy().astype(np.uint8)
            true_bin_np = true_bin.detach().cpu().numpy().astype(np.uint8)

            for b in range(B):
                # Per-image segmentation metrics
                iou_b = masked_iou(pred_probs[b], true_mask[b], buffer_mask[b], threshold=seg_thresh)
                f1_b  = masked_f1(pred_probs[b], true_mask[b], buffer_mask[b], threshold=seg_thresh)

                # Pixel-area metrics per image
                pred_b = pred_bin[b]
                true_b = true_bin[b]
                intersection = (pred_b * true_b).sum()
                pred_area = pred_b.sum()
                true_area = true_b.sum()
                pixel_precision_b = float(intersection / (pred_area + 1e-8)) if float(pred_area) > 0 else 0.0
                pixel_recall_b    = float(intersection / (true_area + 1e-8)) if float(true_area) > 0 else 0.0
                pixel_f1_b = float(2 * pixel_precision_b * pixel_recall_b / (pixel_precision_b + pixel_recall_b + 1e-8))

                # Connected components for centroid-based metrics
                pred_labeled, n_pred = cc_label(pred_bin_np[b] > 0, structure=structure)
                true_labeled, n_true = cc_label(true_bin_np[b] > 0, structure=structure)

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
                rec_b  = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
                f1i_b  = float(2 * prec_b * rec_b / (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0.0
                cerr_b = float(loc_sum_b / loc_count_b) if loc_count_b > 0 else float("nan")

                # Per-image Tree IoU via utility on singleton batch
                tree_stats_b = tree_iou_from_masks(
                    pred_b[None, ...], true_b[None, ...], buffer_mask=buffer_mask[b][None, ...],
                    seg_threshold=seg_thresh, iou_thresh=instance_iou_thresh,
                )

                per_image_metrics.append({
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
                })

            # Aggregate across images in this batch with CIs
            batch_agg = aggregate_epoch_metrics_with_ci(per_image_metrics, confidence=0.95)
            return batch_agg

        logger.info("Configured hybrid loss + sigmoid activation with defaults aligned to eval_pol (instance_iou_threshold=0.4, centroid_max_distance_px=50).")
        return criterion, metrics

    elif conf.loss == "mse":
        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)
            seg_loss = buffer_mask * F.mse_loss(pred[:, 0, :, :], target[:, 0, :, :])
            point_loss = buffer_mask * F.mse_loss(pred[:, 1, :, :], target[:, 1, :, :])
            valid_pixels = buffer_mask.sum() + 1e-8
            return (seg_loss.sum() + point_loss.sum()) / valid_pixels

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            return {
                "mse_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.segment_threshold),
                "mse_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.segment_threshold),
            }
        logger.info("Masked MSE loss configured with buffer weighting.")
        return criterion, metrics

    elif conf.loss == "weighted_dice_loss":
        def criterion(pred, target):
            buffer_mask = target[:, 3, :, :].unsqueeze(1)
            seg_loss = weighted_dice_loss(
                pred[:, 0, :, :], target[:, 0, :, :], buffer_mask=buffer_mask, class_weights=class_weights
            )
            centroid_loss = weighted_dice_loss(
                pred[:, 1, :, :], target[:, 1, :, :], buffer_mask=buffer_mask, class_weights=class_weights
            )
            return seg_loss + centroid_loss

        def metrics(pred, target):
            buffer_mask = target[:, 3, :, :]
            return {
                "iou_segments": masked_iou(pred[:, 0, :, :], target[:, 0, :, :], buffer_mask, threshold=conf.segment_threshold),
                "iou_points": masked_iou(pred[:, 1, :, :], target[:, 1, :, :], buffer_mask, threshold=conf.segment_threshold),
            }
        logger.info("Buffer-weighted Dice loss configured.")
        return criterion, metrics

    else:
        raise ValueError(f"Unsupported loss type: {conf.loss}")