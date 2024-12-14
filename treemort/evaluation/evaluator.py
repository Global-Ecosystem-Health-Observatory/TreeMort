from treemort.utils.iou import IOUCallback
from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def evaluator(model, dataset, num_samples, batch_size, threshold, model_name):
    iou_callback = IOUCallback(
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        threshold=threshold,
        model_name=model_name,
    )

    segmentation_metrics, centroid_metrics = iou_callback.evaluate()

    print("Segmentation Metrics:")
    for key, value in segmentation_metrics.items():
        print(f"{key}: {value:.3f}")

    print("\nCentroid Metrics:")
    for key, value in centroid_metrics.items():
        print(f"{key}: {value:.3f}")
