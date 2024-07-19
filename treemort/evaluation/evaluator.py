from treemort.utils.iou import IOUCallback


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

    print(f" Mean IOU Pixels: {iou_results['mean_iou_pixels']:.3f}")
    print(f" Mean IOU Trees: {iou_results['mean_iou_trees']:.3f}")
