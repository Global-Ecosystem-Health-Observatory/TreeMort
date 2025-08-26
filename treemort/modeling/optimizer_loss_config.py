import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from treemort.utils.loss import weighted_dice_loss, hybrid_loss
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation, proximity_metrics
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment

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

            # Segmentation-level metrics (IoU/F-score with masking)
            iou_segments = masked_iou(pred_probs, true_mask, buffer_mask, threshold=seg_thresh)
            f_score_segments = masked_f1(pred_probs, true_mask, buffer_mask, threshold=seg_thresh)

            # Pixel-level metrics (area-based precision/recall/F1 like eval_pol)
            intersection = (pred_bin * true_bin).sum()
            pred_area = pred_bin.sum()
            true_area = true_bin.sum()
            pixel_precision = intersection / (pred_area + 1e-8)
            pixel_recall = intersection / (true_area + 1e-8)
            pixel_f1_score = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-8)

            # --- Instance-level (centroid-based) metrics, mirroring eval_pol.calculate_centroid_errors ---
            # Use connected components on thresholded masks to obtain instances and their centroids
            B, H, W = pred_bin.shape
            structure = np.ones((3, 3), dtype=bool)  # 8-connectivity

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_pred_instances = 0
            total_true_instances = 0
            loc_sum = 0.0
            loc_count = 0

            # For tree IoU (Hungarian on instance IoUs)
            tree_tp = 0
            tree_fp_total = 0
            tree_fn_total = 0

            pred_bin_np = pred_bin.detach().cpu().numpy().astype(np.uint8)
            true_bin_np = true_bin.detach().cpu().numpy().astype(np.uint8)

            for b in range(B):
                # Connected components
                pred_labeled, n_pred = cc_label(pred_bin_np[b] > 0, structure=structure)
                true_labeled, n_true = cc_label(true_bin_np[b] > 0, structure=structure)

                total_pred_instances += int(n_pred)
                total_true_instances += int(n_true)

                # Skip if empty on either side
                if n_pred == 0 and n_true == 0:
                    continue

                # Compute centroids for centroid-based matching
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

                # Distance matrix (centroid-based)
                if n_pred > 0 and n_true > 0:
                    # Compute pairwise Euclidean distances
                    dists = np.sqrt(((pred_centroids[:, None, :] - true_centroids[None, :, :]) ** 2).sum(axis=2))
                    # Gather candidate pairs within max distance
                    pairs = []
                    for i in range(n_pred):
                        for j in range(n_true):
                            if np.isfinite(dists[i, j]) and dists[i, j] <= max_centroid_dist:
                                pairs.append((dists[i, j], i, j))
                    pairs.sort(key=lambda x: x[0])  # greedy by distance

                    matched_pred = set()
                    matched_true = set()
                    for dist, i, j in pairs:
                        if i not in matched_pred and j not in matched_true:
                            matched_pred.add(i)
                            matched_true.add(j)
                            total_tp += 1
                            loc_sum += float(dist)
                            loc_count += 1
                    total_fp += int(n_pred - len(matched_pred))
                    total_fn += int(n_true - len(matched_true))
                else:
                    total_fp += int(n_pred)
                    total_fn += int(n_true)

                # Tree IoU via Hungarian on instance IoU matrix (like eval_pol.calculate_tree_iou_hungarian)
                if n_pred > 0 and n_true > 0:
                    # Build IoU matrix between instances
                    iou_mat = np.zeros((n_pred, n_true), dtype=float)
                    for i in range(1, n_pred + 1):
                        pred_mask_i = (pred_labeled == i)
                        pred_area_i = pred_mask_i.sum()
                        if pred_area_i == 0:
                            continue
                        for j in range(1, n_true + 1):
                            true_mask_j = (true_labeled == j)
                            inter = np.logical_and(pred_mask_i, true_mask_j).sum()
                            union = pred_area_i + true_mask_j.sum() - inter
                            iou_mat[i - 1, j - 1] = (inter / union) if union > 0 else 0.0

                    # Hungarian on negative IoU (to maximize IoU)
                    row_ind, col_ind = linear_sum_assignment(-iou_mat)
                    batch_tp = 0
                    matched_preds = set()
                    matched_trues = set()
                    for r, c in zip(row_ind, col_ind):
                        if iou_mat[r, c] >= instance_iou_thresh:
                            batch_tp += 1
                            matched_preds.add(r)
                            matched_trues.add(c)
                    batch_fp = n_pred - len(matched_preds)
                    batch_fn = n_true - len(matched_trues)
                    tree_tp += batch_tp
                    tree_fp_total += batch_fp
                    tree_fn_total += batch_fn
                else:
                    tree_fp_total += int(n_pred)
                    tree_fn_total += int(n_true)

            # Instance precision/recall/F1 from centroid-based matching
            instance_precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
            instance_recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
            instance_f1_score = float(2 * instance_precision * instance_recall / (instance_precision + instance_recall)) if (instance_precision + instance_recall) > 0 else 0.0
            centroid_err = float(loc_sum / loc_count) if loc_count > 0 else float("nan")

            # Tree IoU (set-level, IoU-thresholded matching)
            tree_iou = float(tree_tp / (tree_tp + tree_fp_total + tree_fn_total)) if (tree_tp + tree_fp_total + tree_fn_total) > 0 else 0.0

            return {
                "iou_segments": iou_segments,
                "f_score_segments": f_score_segments,
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "pixel_f1_score": pixel_f1_score,
                # Instance-level (centroid-based)
                "instance_precision": instance_precision,
                "instance_recall": instance_recall,
                "instance_f1_score": instance_f1_score,
                "centroid_err": centroid_err,
                "centroid_err_sum": float(loc_sum),
                "centroid_err_count": int(loc_count),
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn),
                "pred_peaks": int(total_pred_instances),
                "true_peaks": int(total_true_instances),
                # Additional metric mirroring eval_pol tree IoU
                "tree_iou": tree_iou,
            }

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