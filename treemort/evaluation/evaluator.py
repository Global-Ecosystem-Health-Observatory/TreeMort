from treemort.utils.iou import IOUCallback
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def evaluator(model, dataset, num_samples, batch_size, threshold, model_name):
    try:
        logger.info("Starting evaluation...")

        iou_callback = IOUCallback(
            model=model,
            dataset=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            threshold=threshold,
            model_name=model_name,
        )

        segmentation_metrics, centroid_metrics = iou_callback.evaluate()

        logger.info("Segmentation Metrics:")
        for key, value in segmentation_metrics.items():
            logger.info(f"{key}: {value:.3f}")

        logger.info("\nCentroid Metrics:")
        for key, value in centroid_metrics.items():
            logger.info(f"{key}: {value:.3f}")

        results = {
            "segmentation_metrics": segmentation_metrics,
            "centroid_metrics": centroid_metrics,
        }

        logger.info("Evaluation completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise