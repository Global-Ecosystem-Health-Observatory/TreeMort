import torch

from collections import defaultdict

from treemort.training.output_processing import process_model_output

from treemort.utils.logger import get_logger
from treemort.utils.metrics import masked_iou, masked_f1, apply_activation, log_metrics

logger = get_logger(__name__)


def evaluator(model, dataloader, num_samples, metrics, conf):
    try:
        logger.info("Starting evaluation...")
        
        seg_metrics = {"iou": 0.0, "f1": 0.0}
        
        model.eval()
        device = next(model.parameters()).device
        total_processed = 0

        test_metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                buffer_mask = labels[:, 3, :, :].unsqueeze(1)  # [B, 1, H, W]
                _, _, h, w = buffer_mask.shape

                logits = process_model_output(model, images, conf.model)
            
                target_mask = labels[:, 0, :, :].unsqueeze(1)  # [B, 1, h, w]
                target_centroid = labels[:, 1, :, :].unsqueeze(1)  # [B, 1, h, w]
                target_hybrid = labels[:, 2, :, :].unsqueeze(1)  # [B, 1, h, w]
                targets = torch.cat([
                    target_mask,        # Channel 0
                    target_centroid,    # Channel 1
                    target_hybrid,      # Channel 2
                    buffer_mask         # Channel 3
                ], dim=1)  # [B, 4, h, w]

                batch_metrics = metrics(logits, targets)

                for key, value in batch_metrics.items():
                    test_metrics[key] += value.item() if torch.is_tensor(value) else value

        for key in test_metrics:
            test_metrics[key] /= len(dataloader)

        log_metrics(test_metrics, "Test")

        return model

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise