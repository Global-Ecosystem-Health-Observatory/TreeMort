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

    iou_results = iou_callback.evaluate()

    logger.info(f"Mean IOU Pixels: {iou_results['mean_iou_pixels']:.3f}")
    logger.info(f"Mean IOU Trees: {iou_results['mean_iou_trees']:.3f}")
    logger.info(f"Mean IOU: {iou_results['mean_iou']:.3f}")
    logger.info(f"Mean Balanced IOU: {iou_results['mean_balanced_iou']:.3f}")
    logger.info(f"Mean Dice Score: {iou_results['mean_dice_score']:.3f}")
    logger.info(f"Mean Adjusted Dice Score: {iou_results['mean_adjusted_dice_score']:.3f}")
    logger.info(f"Mean MCC: {iou_results['mean_mcc']:.3f}")
