import torch

from treemort.training.output_processing import process_model_output

from treemort.utils.logger import get_logger
from treemort.utils.metrics import log_epoch_metrics

logger = get_logger(__name__)


def evaluator(model, dataloader, num_samples, metrics, conf, wandb_run=None):
    try:
        logger.info("Starting evaluation...")
        model.eval()
        device = next(model.parameters()).device

        all_batch_metrics = []

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
                all_batch_metrics.append(batch_metrics)

        summary = log_epoch_metrics(all_batch_metrics, phase="Test", confidence=0.95)

        if wandb_run is not None and getattr(conf, "wandb", False):
            payload = {f"test/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float))}
            payload["phase"] = "test"
            payload["dataset_artifact"] = getattr(conf, "dataset_artifact", None)
            try:
                wandb_run.log(payload)
            except Exception as exc:
                logger.warning(f"Failed to log evaluation metrics to W&B: {exc}")

        return model

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
