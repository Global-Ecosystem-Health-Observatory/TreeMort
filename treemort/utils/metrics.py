import torch
import numpy as np
import math

from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

from treemort.utils.logger import get_logger


# --- Confidence interval utilities ---

def _clean_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[~np.isnan(arr)]


def confidence_interval(values, confidence: float = 0.95):
    """Return (mean, std, ci_low, ci_high) for a list/array, ignoring NaNs.
    Uses normal approximation with z ≈ 1.96 for 95% CI by default.
    """
    arr = _clean_array(values)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    z = 1.96 if abs(confidence - 0.95) < 1e-9 else 1.96
    half_width = z * (std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, mean - half_width, mean + half_width


# Keys intended to be **summed** across batches/images
SUM_KEYS = {
    "tp", "fp", "fn", "pred_peaks", "true_peaks",
    # For centroid error micro-aggregation
    "centroid_err_sum", "centroid_err_count",
    # For Tree IoU micro-aggregation
    "tree_tp", "tree_fp", "tree_fn",
}

# Keys that are typically averaged across batches/images (unless recomputed from sums)
MEAN_KEYS = {
    "iou_segments", "f_score_segments",
    "pixel_precision", "pixel_recall", "pixel_f1_score",
    # Instance rates will be recomputed from sums below, so these are left here for completeness
    "instance_precision", "instance_recall", "instance_f1_score",
    "tree_iou",
}


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


# --- Helper functions for set-level Tree IoU (Hungarian matching on connected components) ---
def _label_cc(binary_np: np.ndarray):
    # 8-connectivity structure
    structure = np.ones((3, 3), dtype=bool)
    labeled, n = label(binary_np > 0, structure=structure)
    return labeled, int(n)

def _tree_iou_counts_single(pred_bin_2d: np.ndarray, true_bin_2d: np.ndarray, iou_thresh: float = 0.4):
    pred_labeled, n_pred = _label_cc(pred_bin_2d)
    true_labeled, n_true = _label_cc(true_bin_2d)

    if n_pred == 0 and n_true == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_true
    if n_true == 0:
        return 0, n_pred, 0

    iou_mat = np.zeros((n_pred, n_true), dtype=float)
    # Build IoU matrix between predicted and true instances
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

    # Hungarian assignment maximizing IoU (minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    tp = 0
    matched_preds = set()
    matched_trues = set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_thresh:
            tp += 1
            matched_preds.add(r)
            matched_trues.add(c)
    fp = n_pred - len(matched_preds)
    fn = n_true - len(matched_trues)
    return int(tp), int(fp), int(fn)

def tree_iou_from_masks(pred_probs: torch.Tensor,
                        true_mask: torch.Tensor,
                        buffer_mask: torch.Tensor = None,
                        seg_threshold: float = 0.5,
                        iou_thresh: float = 0.4):
    """Compute Tree IoU via Hungarian matching on connected components.

    Returns a dict with integer counts and scalar tree_iou.
    """
    if isinstance(pred_probs, torch.Tensor):
        pred_probs_np = pred_probs.detach().cpu().numpy()
    else:
        pred_probs_np = np.asarray(pred_probs)

    if isinstance(true_mask, torch.Tensor):
        true_mask_np = true_mask.detach().cpu().numpy()
    else:
        true_mask_np = np.asarray(true_mask)

    if buffer_mask is not None:
        buffer_np = buffer_mask.detach().cpu().numpy() if isinstance(buffer_mask, torch.Tensor) else np.asarray(buffer_mask)
    else:
        buffer_np = None

    # Binarize with optional buffer
    pred_bin = (pred_probs_np > seg_threshold).astype(np.uint8)
    true_bin = (true_mask_np > seg_threshold).astype(np.uint8)
    if buffer_np is not None:
        m = (buffer_np > 0.5).astype(np.uint8)
        pred_bin = (pred_bin * m).astype(np.uint8)
        true_bin = (true_bin * m).astype(np.uint8)

    if pred_bin.ndim == 2:
        pred_bin = pred_bin[None, ...]
    if true_bin.ndim == 2:
        true_bin = true_bin[None, ...]

    B = pred_bin.shape[0]
    tree_tp = 0
    tree_fp = 0
    tree_fn = 0
    for b in range(B):
        tp, fp, fn = _tree_iou_counts_single(pred_bin[b], true_bin[b], iou_thresh)
        tree_tp += tp
        tree_fp += fp
        tree_fn += fn

    denom = tree_tp + tree_fp + tree_fn
    tree_iou = (tree_tp / denom) if denom > 0 else 0.0
    return {
        "tree_tp": int(tree_tp),
        "tree_fp": int(tree_fp),
        "tree_fn": int(tree_fn),
        "tree_iou": float(tree_iou),
    }


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


from collections import defaultdict

def aggregate_epoch_metrics(batch_metrics_list):
    """Aggregate a sequence of batch-level metric dicts into dataset-level metrics.

    This function ensures count-like fields (TP/FP/FN, peaks) are **summed**
    across batches, avoiding fractional counts caused by naive averaging.
    Instance-level precision/recall/F1 are then recomputed from the summed
    counts (micro-averaging). Other continuous metrics are averaged.

    Parameters
    ----------
    batch_metrics_list : Sequence[dict]
        Iterable of metric dictionaries returned per batch by your metric
        functions (e.g., `configure_loss_and_metrics(...).metrics`).

    Returns
    -------
    dict
        Aggregated metrics where counts are integers (totals) and instance
        rates are recomputed from the global sums.
    """
    if not batch_metrics_list:
        return {}

    acc = defaultdict(float)
    n = 0
    for m in batch_metrics_list:
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            # Convert tensors -> float
            if isinstance(v, torch.Tensor):
                try:
                    v = float(v.detach().cpu().item())
                except Exception:
                    v = float(v.detach().cpu().numpy())
            try:
                v = float(v)
            except Exception:
                # skip non-numeric fields
                continue
            acc[k] += v
        n += 1

    # Final dict
    out = {}

    # 1) Counts: SUM and cast to int
    tp = int(round(acc.get("tp", 0.0)))
    fp = int(round(acc.get("fp", 0.0)))
    fn = int(round(acc.get("fn", 0.0)))
    out["tp"], out["fp"], out["fn"] = tp, fp, fn

    out["pred_peaks"] = int(round(acc.get("pred_peaks", 0.0)))
    out["true_peaks"] = int(round(acc.get("true_peaks", 0.0)))

    # Tree IoU from summed counts if available
    ttp = int(round(acc.get("tree_tp", 0.0)))
    tfp = int(round(acc.get("tree_fp", 0.0)))
    tfn = int(round(acc.get("tree_fn", 0.0)))
    if (ttp + tfp + tfn) > 0:
        out["tree_tp"], out["tree_fp"], out["tree_fn"] = ttp, tfp, tfn
        out["tree_iou"] = float(ttp / (ttp + tfp + tfn))

    # 2) Instance rates: recompute from summed counts (micro-averaging)
    denom_p = tp + fp
    denom_r = tp + fn
    inst_p = (tp / denom_p) if denom_p > 0 else 0.0
    inst_r = (tp / denom_r) if denom_r > 0 else 0.0
    inst_f1 = (2 * inst_p * inst_r / (inst_p + inst_r)) if (inst_p + inst_r) > 0 else 0.0
    out["instance_precision"] = float(inst_p)
    out["instance_recall"] = float(inst_r)
    out["instance_f1_score"] = float(inst_f1)

    # 3) Centroid error: prefer micro-aggregation via sums if present
    loc_sum = acc.get("centroid_err_sum", 0.0)
    loc_cnt = acc.get("centroid_err_count", 0.0)
    if loc_cnt > 0:
        out["centroid_err_sum"] = float(loc_sum)
        out["centroid_err_count"] = int(round(loc_cnt))
        out["centroid_err"] = float(loc_sum / loc_cnt)

    # 4) Other continuous metrics: average over batches
    if n > 0:
        for k in MEAN_KEYS:
            if k in {"instance_precision", "instance_recall", "instance_f1_score"}:
                continue  # already recomputed from sums
            if k in acc:
                out[k] = float(acc[k] / n)

    return out


CI_KEYS = [
    "iou_segments",
    "f_score_segments",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1_score",
    "instance_precision",
    "instance_recall",
    "instance_f1_score",
    "centroid_err",
    "tree_iou",
]


def aggregate_epoch_metrics_with_ci(batch_metrics_list, confidence: float = 0.95):
    """Aggregate metrics and also compute CIs for continuous metrics across batches.

    Returns a dict containing the same fields as `aggregate_epoch_metrics`, plus
    for each key in CI_KEYS: `std_<key>` and `ci_<key>_low`, `ci_<key>_high`.
    """
    base = aggregate_epoch_metrics(batch_metrics_list)

    per_key_values = {k: [] for k in CI_KEYS}
    for m in (batch_metrics_list or []):
        if not isinstance(m, dict):
            continue
        for k in CI_KEYS:
            if k in m:
                v = m[k]
                if isinstance(v, torch.Tensor):
                    try:
                        v = float(v.detach().cpu().item())
                    except Exception:
                        v = float(v.detach().cpu().numpy())
                else:
                    try:
                        v = float(v)
                    except Exception:
                        v = float("nan")
                per_key_values[k].append(v)

    for k, vals in per_key_values.items():
        mean, std, lo, hi = confidence_interval(vals, confidence=confidence)
        if not np.isnan(mean):
            base[f"mean_{k}"] = float(mean)
            base[f"std_{k}"] = float(std)
            base[f"ci_{k}_low"] = float(lo)
            base[f"ci_{k}_high"] = float(hi)

    return base

def log_metrics(metrics, phase):
    logger = get_logger()

    def _fmt_f(x):
        # Safely format floats/tensors
        if isinstance(x, torch.Tensor):
            try:
                x = float(x.detach().cpu().item())
            except Exception:
                x = float(x.detach().cpu().numpy())
        return f"{float(x):.4f}"

    def _fmt_i(x):
        # Safely format integers/tensors
        if isinstance(x, torch.Tensor):
            try:
                x = int(x.detach().cpu().item())
            except Exception:
                x = int(x.detach().cpu().numpy())
        else:
            try:
                x = int(round(float(x)))
            except Exception:
                pass
        return f"{x}"

    # Collect values with safe getters
    def g(key, default=None):
        return metrics.get(key, default)

    # Header
    logger.info(f"{phase} Metrics Summary:\n" + "-" * 34)

    # 1) Segmentation (mask) metrics — pixel-space on binary masks within buffer
    seg_iou = g("iou_segments")
    seg_f1  = g("f_score_segments")
    if seg_iou is not None or seg_f1 is not None:
        logger.info("Segmentation (Pixel-Mask) Metrics:")
        if seg_iou is not None:
            extra = ""
            if all(k in metrics for k in ("ci_iou_segments_low", "ci_iou_segments_high", "std_iou_segments")):
                extra = f" (CI: {metrics['ci_iou_segments_low']:.4f}-{metrics['ci_iou_segments_high']:.4f}, Std: {metrics['std_iou_segments']:.4f})"
            logger.info(f"  Pixel IoU            : {_fmt_f(seg_iou)}{extra}")
        if seg_f1 is not None:
            extra = ""
            if all(k in metrics for k in ("ci_f_score_segments_low", "ci_f_score_segments_high", "std_f_score_segments")):
                extra = f" (CI: {metrics['ci_f_score_segments_low']:.4f}-{metrics['ci_f_score_segments_high']:.4f}, Std: {metrics['std_f_score_segments']:.4f})"
            logger.info(f"  Segment F1-Score     : {_fmt_f(seg_f1)}{extra}")

    # 2) Pixel-area metrics (area-based precision/recall/F1)
    p_prec = g("pixel_precision")
    p_rec  = g("pixel_recall")
    p_f1   = g("pixel_f1_score")
    if any(v is not None for v in (p_prec, p_rec, p_f1)):
        logger.info("Pixel-Area Metrics:")
        if p_prec is not None:
            extra = ""
            if all(k in metrics for k in ("ci_pixel_precision_low", "ci_pixel_precision_high", "std_pixel_precision")):
                extra = f" (CI: {metrics['ci_pixel_precision_low']:.4f}-{metrics['ci_pixel_precision_high']:.4f}, Std: {metrics['std_pixel_precision']:.4f})"
            logger.info(f"  Pixel Precision      : {_fmt_f(p_prec)}{extra}")
        if p_rec is not None:
            extra = ""
            if all(k in metrics for k in ("ci_pixel_recall_low", "ci_pixel_recall_high", "std_pixel_recall")):
                extra = f" (CI: {metrics['ci_pixel_recall_low']:.4f}-{metrics['ci_pixel_recall_high']:.4f}, Std: {metrics['std_pixel_recall']:.4f})"
            logger.info(f"  Pixel Recall         : {_fmt_f(p_rec)}{extra}")
        if p_f1 is not None:
            extra = ""
            if all(k in metrics for k in ("ci_pixel_f1_score_low", "ci_pixel_f1_score_high", "std_pixel_f1_score")):
                extra = f" (CI: {metrics['ci_pixel_f1_score_low']:.4f}-{metrics['ci_pixel_f1_score_high']:.4f}, Std: {metrics['std_pixel_f1_score']:.4f})"
            logger.info(f"  Pixel F1-Score       : {_fmt_f(p_f1)}{extra}")

    # 3) Instance (centroid) metrics
    i_prec = g("instance_precision")
    i_rec  = g("instance_recall")
    i_f1   = g("instance_f1_score")
    if any(v is not None for v in (i_prec, i_rec, i_f1)):
        logger.info("Instance (Centroid) Metrics:")
        if i_prec is not None:
            extra = ""
            if all(k in metrics for k in ("ci_instance_precision_low", "ci_instance_precision_high", "std_instance_precision")):
                extra = f" (CI: {metrics['ci_instance_precision_low']:.4f}-{metrics['ci_instance_precision_high']:.4f}, Std: {metrics['std_instance_precision']:.4f})"
            logger.info(f"  Instance Precision   : {_fmt_f(i_prec)}{extra}")
        if i_rec is not None:
            extra = ""
            if all(k in metrics for k in ("ci_instance_recall_low", "ci_instance_recall_high", "std_instance_recall")):
                extra = f" (CI: {metrics['ci_instance_recall_low']:.4f}-{metrics['ci_instance_recall_high']:.4f}, Std: {metrics['std_instance_recall']:.4f})"
            logger.info(f"  Instance Recall      : {_fmt_f(i_rec)}{extra}")
        if i_f1 is not None:
            extra = ""
            if all(k in metrics for k in ("ci_instance_f1_score_low", "ci_instance_f1_score_high", "std_instance_f1_score")):
                extra = f" (CI: {metrics['ci_instance_f1_score_low']:.4f}-{metrics['ci_instance_f1_score_high']:.4f}, Std: {metrics['std_instance_f1_score']:.4f})"
            logger.info(f"  Instance F1-Score    : {_fmt_f(i_f1)}{extra}")

    # 4) Tree-level metric
    t_iou = g("tree_iou")
    if t_iou is not None:
        logger.info("Set-Level Metric:")
        extra = ""
        if all(k in metrics for k in ("ci_tree_iou_low", "ci_tree_iou_high", "std_tree_iou")):
            extra = f" (CI: {metrics['ci_tree_iou_low']:.4f}-{metrics['ci_tree_iou_high']:.4f}, Std: {metrics['std_tree_iou']:.4f})"
        logger.info(f"  Tree IoU             : {_fmt_f(t_iou)}{extra}")

    # 5) Counts
    counts = []
    for key, label in (
        ("tp", "Instance TP"),
        ("fp", "Instance FP"),
        ("fn", "Instance FN"),
        ("pred_peaks", "Pred Peaks"),
        ("true_peaks", "True Peaks"),
        ("tree_tp", "Tree TP"),
        ("tree_fp", "Tree FP"),
        ("tree_fn", "Tree FN"),
    ):
        if key in metrics:
            counts.append((label, _fmt_i(metrics[key])))
    if counts:
        logger.info("Counts:")
        for label, val in counts:
            logger.info(f"  {label:<20}: {val}")

    # 6) Centroid localization error (if available)
    cerr = g("centroid_err")
    if cerr is not None:
        # only log real numbers (ignore NaN)
        try:
            val = float(cerr.detach().cpu().item()) if isinstance(cerr, torch.Tensor) else float(cerr)
            if not (isinstance(val, float) and (math.isnan(val) or np.isnan(val))):
                logger.info(f"Centroid Error         : {_fmt_f(val)}")
        except Exception:
            pass
    if "centroid_err_count" in metrics:
        try:
            cnt = _fmt_i(metrics["centroid_err_count"])
            logger.info(f"  Centroid Matches     : {cnt}")
        except Exception:
            pass