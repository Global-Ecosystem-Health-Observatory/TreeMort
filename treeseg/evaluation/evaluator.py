from treeseg.utils.iou import IOUCallback


def evaluator(model, dataset, num_samples, batch_size):
    iou_callback = IOUCallback(
        model, dataset=dataset, num_samples=num_samples, batch_size=batch_size
    )

    iou_results = iou_callback.evaluate()

    print(f"IOU Pixels: {iou_results['iou_pixels']:.3f}")
    print(f"IOU Trees: {iou_results['iou_trees']:.3f}")
