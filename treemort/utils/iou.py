import torch
import numpy as np
from tqdm import tqdm
from scipy import ndimage

from transformers import MaskFormerImageProcessor, AutoImageProcessor

class IOUCallback:
    def __init__(self, model, dataset, num_samples, batch_size, threshold, model_name, image_processor=None):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_name = model_name
        self.device = next(model.parameters()).device  # Get the device of the model
        self.image_processor = image_processor

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        pixel_ious = []
        tree_ious = []
            
        total_batches = self.num_samples // self.batch_size

        with tqdm(total=total_batches, desc="Evaluating") as pbar:

            with torch.no_grad():
                for images, labels in self.dataset:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    if self.model_name == "maskformer":
                        outputs = self.model(images)

                        target_sizes = [(image.shape[1], image.shape[2]) for image in images]
                        predictions = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                        predictions = torch.stack([prediction.unsqueeze(0) for prediction in predictions] , dim=0)

                    elif self.model_name == "detr":
                        outputs = self.model(images)

                        target_sizes = [(image.shape[1], image.shape[2]) for image in images]
                        predictions = self.image_processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
                        predictions = torch.stack([prediction['segmentation'].unsqueeze(0) for prediction in predictions], dim=0)

                    elif self.model_name == "beit":
                        outputs = self.model(images)

                        target_sizes = [(image.shape[1], image.shape[2]) for image in images]
                        predictions = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                        predictions = torch.stack([prediction.unsqueeze(0) for prediction in predictions], dim=0).float()

                    else:
                        predictions = self.model(images)

                    predictions = predictions.cpu().numpy()
                    labels = labels.cpu().numpy()

                    for i in range(len(predictions)):
                        y_pred = predictions[i]
                        y_true = labels[i]

                        y_pred_binary = np.squeeze(y_pred > self.threshold)
                        y_true_binary = np.squeeze(y_true > self.threshold)

                        # Pixel-wise IoU
                        tp_pixels = np.sum(np.logical_and(y_pred_binary, y_true_binary))
                        fp_pixels = np.sum(
                            np.logical_and(y_pred_binary, np.logical_not(y_true_binary))
                        )
                        fn_pixels = np.sum(
                            np.logical_and(np.logical_not(y_pred_binary), y_true_binary)
                        )

                        if (tp_pixels + fp_pixels + fn_pixels) == 0:
                            iou_pixels = 1.0 if np.sum(y_true_binary) == 0 else 0.0
                        else:
                            iou_pixels = tp_pixels / (tp_pixels + fp_pixels + fn_pixels)
                        pixel_ious.append(iou_pixels)

                        # Tree-wise IoU
                        labeled_array_pred, num_features_pred = ndimage.label(
                            y_pred_binary
                        )
                        labeled_array_true, num_features_true = ndimage.label(
                            y_true_binary
                        )
                        predicted_contour_numbers = np.unique(labeled_array_pred)
                        true_contour_numbers = np.unique(labeled_array_true)

                        tp_trees, fp_trees, fn_trees = 0, 0, 0

                        for predicted_contour_number in predicted_contour_numbers:
                            if predicted_contour_number == 0:
                                continue
                            predicted_contour_mask = (
                                labeled_array_pred == predicted_contour_number
                            )
                            prediction_exists = y_true_binary[
                                predicted_contour_mask
                            ].any()
                            if prediction_exists:
                                tp_trees += 1
                            else:
                                fp_trees += 1

                        for true_contour_number in true_contour_numbers:
                            if true_contour_number == 0:
                                continue
                            true_contour_mask = (
                                labeled_array_true == true_contour_number
                            )
                            prediction_exists = y_pred_binary[true_contour_mask].any()
                            if not prediction_exists:
                                fn_trees += 1

                        if (tp_trees + fp_trees + fn_trees) == 0:
                            iou_trees = 1.0 if num_features_true == 0 else 0.0
                        else:
                            iou_trees = tp_trees / (tp_trees + fp_trees + fn_trees)
                        tree_ious.append(iou_trees)

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_postfix(iterations_left=total_batches - pbar.n)

        mean_iou_pixels = np.mean([iou for iou in pixel_ious if iou != -1])
        mean_iou_trees = np.mean([iou for iou in tree_ious if iou != -1])

        return {"mean_iou_pixels": mean_iou_pixels, "mean_iou_trees": mean_iou_trees}
