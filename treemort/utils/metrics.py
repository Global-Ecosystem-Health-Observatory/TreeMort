import torch
import numpy as np
import math

from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter
from scipy.ndimage import label

from treemort.utils.logger import get_logger


def iou_score(pred_probs, target, threshold=0.5):
    pred = (pred_probs > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return torch.tensor(0.0)

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def f_score(pred_probs, target, threshold=0.5, beta=1):
    pred = (pred_probs > threshold).float()
    target = (target > threshold).float()

    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()

    f_score = ((1 + beta**2) * tp + 1e-6) / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-6)
    return f_score


def masked_mse(pred, target, mask):
    return ((pred - target)**2 * mask).sum() / (mask.sum() + 1e-8)


def masked_iou(pred_probs, target, buffer_mask, threshold=0.5):
    pred = (pred_probs > threshold).float()
    target = (target > threshold).float()
    
    pred = pred * buffer_mask
    target = target * buffer_mask
    
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    
    if union < 1e-8:
        return torch.tensor(0.0, device=pred_probs.device)
    
    return intersection / union


def masked_f1(pred_probs, target, buffer_mask, threshold=0.5):
    pred = (pred_probs > threshold).float()
    target = (target > threshold).float()

    pred = pred * buffer_mask
    target = target * buffer_mask
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


def extract_centroids_from_heatmap(heatmap, threshold=0.5, min_distance=5):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    else:
        heatmap = np.asarray(heatmap)
    
    if heatmap.ndim != 3:
        raise ValueError(f"Expected input with shape (batch, height, width), but got {heatmap.shape}")

    B, H, W = heatmap.shape
    footprint = np.ones((min_distance, min_distance), dtype=bool)
    
    all_centroids = []
    for i in range(B):
        max_filter = maximum_filter(heatmap[i], footprint=footprint, mode='constant')
        peaks = (heatmap[i] == max_filter) & (heatmap[i] > threshold)
        centroids = np.column_stack(np.where(peaks))
        all_centroids.append(centroids.astype(np.float32) if centroids.size > 0 else np.empty((0, 2), dtype=np.float32))

    return all_centroids


def proximity_metrics(
    pred_centroid_map,
    true_centroid_map,
    buffer_mask=None,
    proximity_threshold=5,
    threshold=0.1,
    min_distance=5,
    true_threshold=None,
):
    # Ensure input arrays are numpy arrays with shape (B, H, W)
    pred_centroid_map = pred_centroid_map.detach().cpu().numpy() if isinstance(pred_centroid_map, torch.Tensor) else np.asarray(pred_centroid_map)
    true_centroid_map = true_centroid_map.detach().cpu().numpy() if isinstance(true_centroid_map, torch.Tensor) else np.asarray(true_centroid_map)
    if pred_centroid_map.ndim == 2:
        pred_centroid_map = pred_centroid_map[None, ...]
    if true_centroid_map.ndim == 2:
        true_centroid_map = true_centroid_map[None, ...]
    if buffer_mask is not None:
        buffer_mask = buffer_mask.detach().cpu().numpy() if isinstance(buffer_mask, torch.Tensor) else np.asarray(buffer_mask)
        if buffer_mask.ndim == 2:
            buffer_mask = buffer_mask[None, ...]
        buffer_mask = (buffer_mask > 0.5).astype(buffer_mask.dtype)
    thr_true = threshold if true_threshold is None else true_threshold

    B = pred_centroid_map.shape[0]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_peaks = 0
    total_true_peaks = 0
    matched_distances = []

    def extract_centroids_single(heatmap2d, threshold, min_distance):
        # heatmap2d: (H, W)
        footprint = np.ones((min_distance, min_distance), dtype=bool)
        max_filter = maximum_filter(heatmap2d, footprint=footprint, mode='constant')
        peaks = (heatmap2d == max_filter) & (heatmap2d > threshold)
        centroids = np.column_stack(np.where(peaks))
        return centroids.astype(np.float32) if centroids.size > 0 else np.empty((0, 2), dtype=np.float32)

    for i in range(B):
        pred_map = pred_centroid_map[i]
        true_map = true_centroid_map[i]
        if buffer_mask is not None:
            mask = buffer_mask[i]
            pred_map = pred_map * mask
            true_map = true_map * mask
        pred_centroids = extract_centroids_single(pred_map, threshold, min_distance)
        true_centroids = extract_centroids_single(true_map, thr_true, min_distance)
        total_pred_peaks += int(pred_centroids.shape[0])
        total_true_peaks += int(true_centroids.shape[0])

        if pred_centroids.shape[0] == 0 or true_centroids.shape[0] == 0:
            tp = 0
            fp = len(pred_centroids)
            fn = len(true_centroids)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            continue
        # Pairwise distances
        distances = cdist(pred_centroids, true_centroids)
        # Find all pairs with distance <= proximity_threshold
        pairs = []
        for p_idx in range(distances.shape[0]):
            for t_idx in range(distances.shape[1]):
                if distances[p_idx, t_idx] <= proximity_threshold:
                    pairs.append((distances[p_idx, t_idx], p_idx, t_idx))
        # Sort pairs by distance (greedy matching)
        pairs.sort()
        matched_pred = set()
        matched_true = set()
        matched_this = []
        for dist, p_idx, t_idx in pairs:
            if p_idx not in matched_pred and t_idx not in matched_true:
                matched_pred.add(p_idx)
                matched_true.add(t_idx)
                matched_this.append(dist)
        tp = len(matched_pred)
        fp = len(pred_centroids) - tp
        fn = len(true_centroids) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_distances.extend(matched_this)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    loc_count = len(matched_distances)
    if loc_count > 0:
        loc_sum = float(np.sum(matched_distances))
        loc_error = float(loc_sum / loc_count)
    else:
        loc_sum = 0.0
        loc_error = 0.0  # ignore NaNs by using 0 for batches with no matches

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "localization_error": loc_error,
        "localization_error_sum": loc_sum,
        "localization_count": int(loc_count),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "pred_peaks": int(total_pred_peaks),
        "true_peaks": int(total_true_peaks),
    }


def apply_activation(logits, activation="sigmoid"):
    activations = {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid
    }

    if activation not in activations:
        raise ValueError(f"Unsupported activation type: {activation}")

    return activations[activation](logits)


def log_metrics(metrics, phase):
    logger = get_logger()

    # Segmentation metrics: IoU, F-Score
    seg_metrics = {k: v for k, v in metrics.items() if "segments" in k}
    seg_order = ["iou_segments", "f_score_segments"]
    if seg_metrics:
        logger.info(f"{phase} Segmentation Metrics:")
        seg_name_map = {
            "iou_segments": "IoU",
            "f_score_segments": "F-Score",
        }
        for key in seg_order:
            if key in seg_metrics:
                display_name = seg_name_map.get(key, key.replace("_segments", "").replace("_", " ").title())
                logger.info(f"  {display_name}: {seg_metrics[key]:.4f}")

    # Centroid metrics: IoU, F-Score
    cent_metrics = {k: v for k, v in metrics.items() if "points" in k}
    cent_order = ["iou_points", "f_score_points"]
    if cent_metrics:
        logger.info(f"{phase} Centroid Metrics:")
        cent_name_map = {
            "iou_points": "IoU",
            "f_score_points": "F-Score",
        }
        for key in cent_order:
            if key in cent_metrics:
                display_name = cent_name_map.get(key, key.replace("_points", "").replace("_", " ").title())
                logger.info(f"  {display_name}: {cent_metrics[key]:.4f}")

    # Pixel-level metrics: Pixel Precision, Pixel Recall, Pixel F1-Score
    pixel_order = ["pixel_precision", "pixel_recall", "pixel_f1_score"]
    pixel_metrics = {k: v for k, v in metrics.items() if k in pixel_order}
    if pixel_metrics:
        logger.info(f"{phase} Pixel-level Metrics:")
        pixel_name_map = {
            "pixel_precision": "Pixel Precision",
            "pixel_recall": "Pixel Recall",
            "pixel_f1_score": "Pixel F1-Score",
        }
        for key in pixel_order:
            if key in pixel_metrics:
                display_name = pixel_name_map.get(key, key.replace("pixel_", "").replace("_", " ").title())
                logger.info(f"  {display_name}: {pixel_metrics[key]:.4f}")

    # Instance-level metrics: Instance Precision, Instance Recall, Instance F1-Score
    inst_order = ["instance_precision", "instance_recall", "instance_f1_score"]
    inst_metrics = {k: v for k, v in metrics.items() if k in inst_order}
    if inst_metrics:
        logger.info(f"{phase} Instance Metrics:")
        inst_name_map = {
            "instance_precision": "Instance Precision",
            "instance_recall": "Instance Recall",
            "instance_f1_score": "Instance F1-Score",
        }
        for key in inst_order:
            if key in inst_metrics:
                display_name = inst_name_map.get(key, key.replace("_instance", "").replace("_", " ").title())
                logger.info(f"  {display_name}: {inst_metrics[key]:.4f}")

    # Instance counts (if provided)
    counts_map = {
        "tp": "Instance TP",
        "fp": "Instance FP",
        "fn": "Instance FN",
        "pred_peaks": "Pred Peaks",
        "true_peaks": "True Peaks",
    }
    have_any = any(k in metrics for k in counts_map)
    if have_any:
        for k, label in counts_map.items():
            if k in metrics:
                v = metrics[k]
                if isinstance(v, torch.Tensor):
                    try:
                        v = int(v.detach().cpu().item())
                    except Exception:
                        v = int(v)
                logger.info(f"  {label}: {v}")

    # Log Centroid Error after instance metrics (ignore NaNs)
    if "centroid_err" in metrics:
        val = metrics["centroid_err"]
        # Convert tensors to plain float if needed
        if isinstance(val, torch.Tensor):
            try:
                val = float(val.detach().cpu().item())
            except Exception:
                val = float(val.detach().cpu().numpy())
        # Only log if the value is a real number (not NaN)
        if isinstance(val, (float, int)) and not (isinstance(val, float) and (math.isnan(val) or np.isnan(val))):
            logger.info(f"{phase} Centroid Error: {val:.4f}")

    if "centroid_err_count" in metrics:
        try:
            cnt = int(metrics["centroid_err_count"]) if not isinstance(metrics["centroid_err_count"], torch.Tensor) else int(metrics["centroid_err_count"].detach().cpu().item())
            logger.info(f"  Centroid Matches: {cnt}")
        except Exception:
            pass

    # Proximity metrics: log in alphabetical order, but ensure display name consistency
    prox_metrics = {k: v for k, v in metrics.items() if "proximity" in k}
    if prox_metrics:
        logger.info(f"{phase} Proximity Metrics:")
        prox_keys = sorted(prox_metrics.keys())
        for key in prox_keys:
            display_name = key.replace("_", " ").title()
            logger.info(f"  {display_name}: {prox_metrics[key]:.4f}")