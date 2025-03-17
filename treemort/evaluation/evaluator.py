import torch

from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation

logger = get_logger(__name__)


def evaluator(model, dataloader, num_samples, conf):
    try:
        logger.info("Starting evaluation...")
        
        seg_metrics = {"iou": 0.0, "f1": 0.0}
        
        model.eval()
        device = next(model.parameters()).device
        total_processed = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                
                images = images.to(device)
                labels = labels.to(device)
                
                preds = model(images)
    
                for i in range(preds.shape[0]):
                    if total_processed >= num_samples:
                        break
                    
                    buffer_mask = labels[i, 3, :, :]  # [H, W]
                    h, w = buffer_mask.shape

                    pred_mask = apply_activation(preds[0, 0], activation=conf.activation) * buffer_mask
                    true_mask = labels[i, 0] * buffer_mask
                    
                    seg_metrics["iou"] += masked_iou(pred_mask, true_mask, buffer_mask, threshold=conf.segment_threshold)
                    seg_metrics["f1"] += masked_f1(pred_mask, true_mask, buffer_mask, threshold=conf.segment_threshold)
                    
                    total_processed += 1

        if total_processed > 0:
            seg_metrics = {k: v/total_processed for k,v in seg_metrics.items()}
        else:
            logger.warning("No samples were processed during evaluation.")
            seg_metrics = {k: 0.0 for k in seg_metrics.keys()}

        logger.info("Segmentation Metrics:")
        for key, value in seg_metrics.items():
            logger.info(f"{key}: {value:.3f}")

        logger.info(f"Evaluation completed on {total_processed} samples.")
        return {"segmentation_metrics": seg_metrics}

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise