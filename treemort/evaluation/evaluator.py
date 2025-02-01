import torch

from treemort.utils.iou import IOUCallback
from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, proximity_metrics

logger = get_logger(__name__)


def evaluator(model, dataloader, num_samples, batch_size, threshold, model_name):
    try:
        logger.info("Starting evaluation...")
        
        seg_metrics = {"iou": 0.0, "f1": 0.0}
        centroid_metrics = {"precision": 0.0, "recall": 0.0, 
                          "f1_score": 0.0, "localization_error": 0.0}
        
        model.eval()
        device = next(model.parameters()).device
        total_processed = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                
                images = images.to(device)
                labels = labels.to(device)
                
                preds = model(images)  # [B,3,H,W]
    
                for i in range(preds.shape[0]):
                    if total_processed >= num_samples:
                        break
                    
                    buffer_mask = labels[i, 3, :, :]  # [H, W]
                    h, w = buffer_mask.shape
                    
                    cropped_preds = torch.cat([
                        center_crop(preds[i:i+1, 0:1], (h, w)),  # Mask
                        center_crop(preds[i:i+1, 1:2], (h, w)),  # Centroid
                        center_crop(preds[i:i+1, 2:3], (h, w))   # Hybrid
                    ], dim=1)  # [1,3,h,w]
                    
                    # cropped_labels = center_crop(labels[i:i+1], (h, w))  # [1,4,h,w]
                
                    pred_mask = torch.sigmoid(cropped_preds[0, 0]) * buffer_mask
                    true_mask = labels[i, 0] * buffer_mask
                    
                    pred_centroid = torch.sigmoid(cropped_preds[0, 1]) * buffer_mask
                    true_centroid = labels[i, 1] * buffer_mask
                    
                    seg_metrics["iou"] += masked_iou(pred_mask, true_mask, buffer_mask, threshold)
                    seg_metrics["f1"] += masked_f1(pred_mask, true_mask, buffer_mask, threshold)
                    
                    centroid_stats = proximity_metrics(
                        pred_centroid.unsqueeze(0),
                        true_centroid.unsqueeze(0),
                        buffer_mask=buffer_mask.unsqueeze(0),
                        proximity_threshold=5,
                        threshold=0.1
                    )
                    for k in centroid_stats:
                        centroid_metrics[k] += centroid_stats[k]
                    
                    total_processed += 1

        seg_metrics = {k: v/total_processed for k,v in seg_metrics.items()}
        centroid_metrics = {k: v/total_processed for k,v in centroid_metrics.items()}

        logger.info("Segmentation Metrics:")
        for key, value in seg_metrics.items():
            logger.info(f"{key}: {value:.3f}")

        logger.info("\nCentroid Metrics:")
        for key, value in centroid_metrics.items():
            logger.info(f"{key}: {value:.3f}")

        logger.info(f"Evaluation completed on {total_processed} samples.")
        return {
            "segmentation_metrics": seg_metrics,
            "centroid_metrics": centroid_metrics
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def center_crop(tensor, target_size):
    """Center crops 4D tensor (B,C,H,W) to target_size (H,W)"""
    _, _, h, w = tensor.size()
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[..., i:i+th, j:j+tw]
