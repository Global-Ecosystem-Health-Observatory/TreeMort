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


def masked_mse(pred, target, mask):
    return ((pred - target)**2 * mask).sum() / (mask.sum() + 1e-8)


def masked_iou(pred_logits, target, buffer_mask, threshold=0.5):
    pred_probs = torch.sigmoid(pred_logits)
    pred = (pred_probs > threshold).float()
    
    pred = pred * buffer_mask
    target = target * buffer_mask
    
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    
    if union < 1e-8:
        return torch.tensor(0.0, device=pred.device)
    
    return intersection / union

def masked_f1(pred_logits, target, buffer_mask, threshold=0.5):
    pred_probs = torch.sigmoid(pred_logits)
    pred = (pred_probs > threshold).float()
    
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
    footprint_shape = tuple([min_distance] * heatmap.ndim)
    peaks = (heatmap == maximum_filter(heatmap, footprint=np.ones(footprint_shape)))
    
    peaks = peaks & (heatmap > threshold)
    
    centroids = np.argwhere(peaks)
    return [tuple(coord) for coord in centroids]


def proximity_metrics(pred_centroids_map, true_centroids_map, buffer_mask=None, 
                     proximity_threshold=5, threshold=0.1, min_distance=5):
    pred_centroids_map = pred_centroids_map.detach().cpu().numpy() if isinstance(pred_centroids_map, torch.Tensor) else pred_centroids_map
    true_centroids_map = true_centroids_map.detach().cpu().numpy() if isinstance(true_centroids_map, torch.Tensor) else true_centroids_map
    buffer_mask = buffer_mask.detach().cpu().numpy() if isinstance(buffer_mask, torch.Tensor) else buffer_mask

    if buffer_mask is not None:
        pred_centroids_map = pred_centroids_map * buffer_mask
        true_centroids_map = true_centroids_map * buffer_mask

    pred_centroids = extract_centroids_from_heatmap(pred_centroids_map, threshold, min_distance)
    true_centroids = extract_centroids_from_heatmap(true_centroids_map, threshold, min_distance)

    if len(pred_centroids) == 0 and len(true_centroids) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "localization_error": 0.0
        }

    if len(pred_centroids) == 0 or len(true_centroids) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "localization_error": float("inf")
        }

    distances = cdist(pred_centroids, true_centroids)
    
    matches = distances <= proximity_threshold
    matched_pred = set()
    matched_true = set()

    for p_idx, row in enumerate(matches):
        for t_idx, match in enumerate(row):
            if match and p_idx not in matched_pred and t_idx not in matched_true:
                matched_pred.add(p_idx)
                matched_true.add(t_idx)

    tp = len(matched_pred)
    fp = len(pred_centroids) - tp
    fn = len(true_centroids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_distances = [distances[p,t] for p,t in zip(matched_pred, matched_true)]
    loc_error = np.mean(matched_distances) if matched_distances else float('inf')

    return {
        "precision_points": precision,
        "recall_points": recall,
        "f1_score_points": f1,
        "localization_error": loc_error
    }