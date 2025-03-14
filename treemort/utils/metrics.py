import torch
import numpy as np

from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter


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


def proximity_metrics(pred_centroid_map, true_centroid_map, buffer_mask=None,
                      proximity_threshold=5, threshold=0.1, min_distance=5):
    
    pred_centroid_map = pred_centroid_map.detach().cpu().numpy() if isinstance(pred_centroid_map, torch.Tensor) else np.asarray(pred_centroid_map)
    true_centroid_map = true_centroid_map.detach().cpu().numpy() if isinstance(true_centroid_map, torch.Tensor) else np.asarray(true_centroid_map)
    
    if buffer_mask is not None:
        buffer_mask = buffer_mask.detach().cpu().numpy() if isinstance(buffer_mask, torch.Tensor) else np.asarray(buffer_mask)
        pred_centroid_map *= buffer_mask
        true_centroid_map *= buffer_mask

    def safe_extract(heatmap):
        centroids_list = extract_centroids_from_heatmap(heatmap, threshold, min_distance)  # List of (N_i, 2) arrays
        centroids = np.concatenate(centroids_list, axis=0) if centroids_list else np.empty((0, 2), dtype=np.float32)        
        return centroids

    pred_centroids = safe_extract(pred_centroid_map)
    true_centroids = safe_extract(true_centroid_map)

    if pred_centroids.shape[0] == 0 or true_centroids.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "localization_error": float("inf")
        }

    assert pred_centroids.shape[1] == 2, f"Pred centroids shape: {pred_centroids.shape}"
    assert true_centroids.shape[1] == 2, f"True centroids shape: {true_centroids.shape}"
    
    distances = cdist(pred_centroids, true_centroids)

    matches = distances <= proximity_threshold
    matched_pred = set()
    matched_true = set()

    for p_idx, p_row in enumerate(matches):
        for t_idx, is_match in enumerate(p_row):
            if is_match and p_idx not in matched_pred and t_idx not in matched_true:
                matched_pred.add(p_idx)
                matched_true.add(t_idx)

    tp = len(matched_pred)
    fp = len(pred_centroids) - tp
    fn = len(true_centroids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_dists = [distances[p, t] for p, t in zip(matched_pred, matched_true)]
    loc_error = np.mean(matched_dists) if matched_dists else float('inf')

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "localization_error": loc_error
    }


def apply_activation(logits, activation="sigmoid"):
    if activation == "tanh":
        probs = torch.tanh(logits)
    elif activation == "sigmoid":
        probs = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unsupported activation type: {activation}")

    return probs