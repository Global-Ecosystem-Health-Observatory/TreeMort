import torch
import numpy as np

from tqdm import tqdm
from scipy import ndimage


class IOUCallback:
    def __init__(self, model, dataset, num_samples, batch_size, threshold, model_name, image_processor=None):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_name = model_name
        self.image_processor = image_processor
        self.device = next(model.parameters()).device  # Get the device of the model

    def evaluate(self):
        self.model.eval()
        pixel_ious, tree_ious = [], []

        with tqdm(total=self.num_samples, desc="Evaluating") as pbar:
            with torch.no_grad():
                for images, labels in self.dataset:
                    images, labels = images.to(self.device), labels.to(self.device)
                    predictions = self._get_predictions(images, labels)
                    predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()

                    for i in range(len(predictions)):
                        y_pred, y_true = predictions[i], labels[i]
                        iou_pixels = self._calculate_pixel_iou(y_pred, y_true)
                        pixel_ious.append(iou_pixels)
                        iou_trees = self._calculate_tree_iou(y_pred, y_true)
                        tree_ious.append(iou_trees)

                    pbar.update(1)
                    pbar.set_postfix(iterations_left=self.num_samples - pbar.n)

        return self._compute_mean_ious(pixel_ious, tree_ious)

    def _get_predictions(self, images, labels):
        outputs = self.model(images)
        target_sizes = [(label.shape[1], label.shape[2]) for label in labels]

        if self.model_name == "maskformer":
            predictions = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        elif self.model_name == "detr":
            predictions = self.image_processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
            predictions = torch.stack([prediction["segmentation"].unsqueeze(0) for prediction in predictions], dim=0)
        elif self.model_name == "beit":
            predictions = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        elif self.model_name == "dinov2":
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).unsqueeze(1).float()
        else:
            predictions = outputs

        return torch.stack([prediction.unsqueeze(0) for prediction in predictions], dim=0).float()

    def _calculate_pixel_iou(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_pixels = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_pixels = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_pixels = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))

        if (tp_pixels + fp_pixels + fn_pixels) == 0:
            return 1.0 if np.sum(y_true_binary) == 0 else 0.0

        return tp_pixels / (tp_pixels + fp_pixels + fn_pixels)

    def _calculate_tree_iou(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        labeled_array_pred, num_features_pred = ndimage.label(y_pred_binary)
        labeled_array_true, num_features_true = ndimage.label(y_true_binary)

        tp_trees, fp_trees, fn_trees = 0, 0, 0

        for contour_number in np.unique(labeled_array_pred):
            if contour_number == 0:
                continue
            mask = labeled_array_pred == contour_number
            if y_true_binary[mask].any():
                tp_trees += 1
            else:
                fp_trees += 1

        for contour_number in np.unique(labeled_array_true):
            if contour_number == 0:
                continue
            mask = labeled_array_true == contour_number
            if not y_pred_binary[mask].any():
                fn_trees += 1

        if (tp_trees + fp_trees + fn_trees) == 0:
            return 1.0 if num_features_true == 0 else 0.0

        return tp_trees / (tp_trees + fp_trees + fn_trees)

    def _compute_mean_ious(self, pixel_ious, tree_ious):
        mean_iou_pixels = np.mean([iou for iou in pixel_ious if iou != -1])
        mean_iou_trees = np.mean([iou for iou in tree_ious if iou != -1])
        return {"mean_iou_pixels": mean_iou_pixels, "mean_iou_trees": mean_iou_trees}
