import torch
import numpy as np

from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, maximum_filter


def iou_score(pred_probs, target, threshold=0.5):
    pred_binary = (pred_probs > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return torch.tensor(0.0)

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def f_score(pred_probs, target, threshold=0.5, beta=1):
    pred = (pred_probs > threshold).float()
    target = target.float()

    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()

    f_score = ((1 + beta**2) * tp + 1e-6) / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-6)
    return f_score


def extract_centroids_from_heatmap(heatmap, threshold=0.5, min_distance=5):
    footprint_shape = tuple([min_distance] * heatmap.ndim)
    peaks = (heatmap == maximum_filter(heatmap, footprint=np.ones(footprint_shape)))
    
    peaks = peaks & (heatmap > threshold)
    
    centroids = np.argwhere(peaks)
    return [tuple(coord) for coord in centroids]


def proximity_metrics(pred_centroids_map, true_centroids_map, proximity_threshold=5, threshold=0.5, min_distance=5):
    pred_centroids_map = pred_centroids_map.detach().cpu().numpy() if isinstance(pred_centroids_map, torch.Tensor) else pred_centroids_map
    true_centroids_map = true_centroids_map.detach().cpu().numpy() if isinstance(true_centroids_map, torch.Tensor) else true_centroids_map

    pred_centroids = extract_centroids_from_heatmap(pred_centroids_map, threshold, min_distance)
    true_centroids = extract_centroids_from_heatmap(true_centroids_map, threshold, min_distance)

    if len(pred_centroids) == 0 and len(true_centroids) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "localization_error": 0.0}

    if len(pred_centroids) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "localization_error": float("inf")}

    if len(true_centroids) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "localization_error": float("inf")}

    distances = cdist(pred_centroids, true_centroids)

    matches = distances <= proximity_threshold

    matched_pred_indices = set()
    matched_true_indices = set()

    for pred_idx, row in enumerate(matches):
        for true_idx, is_match in enumerate(row):
            if is_match and pred_idx not in matched_pred_indices and true_idx not in matched_true_indices:
                matched_pred_indices.add(pred_idx)
                matched_true_indices.add(true_idx)

    tp = len(matched_pred_indices)
    fp = len(pred_centroids) - tp
    fn = len(true_centroids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_distances = []
    for pred_idx, row in enumerate(matches):
        for true_idx, is_match in enumerate(row):
            if is_match and pred_idx in matched_pred_indices and true_idx in matched_true_indices:
                matched_distances.append(distances[pred_idx, true_idx])

    localization_error = np.mean(matched_distances) if matched_distances else float("inf")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "localization_error": localization_error,
    }