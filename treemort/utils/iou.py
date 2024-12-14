import torch
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist


class IOUCallback:
    def __init__(self, model, dataset, num_samples, batch_size, threshold, model_name):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_name = model_name
        self.device = next(model.parameters()).device

    def evaluate(self):
        self.model.eval()

        segmentation_metrics_acc = {
            "pixel_ious": [],
            "tree_ious": [],
            "mean_ious": [],
            "balanced_ious": [],
            "dice_scores": [],
            "adjusted_dice_scores": [],
            "mcc_scores": [],
        }
        centroid_metrics_acc = {"precision": [], "recall": [], "f1_score": []}

        with tqdm(total=self.num_samples, desc="Evaluating") as pbar:
            with torch.no_grad():
                for images, labels in self.dataset:
                    images, labels = images.to(self.device), labels.to(self.device)

                    pred_segmentation, pred_centroids = self._get_predictions(images)

                    true_segmentation = labels[:, 0, :, :]
                    true_centroids = labels[:, 1, :, :]

                    for i in range(len(pred_segmentation)):
                        y_pred_seg = pred_segmentation[i].cpu().numpy()
                        y_true_seg = true_segmentation[i].cpu().numpy()

                        segmentation_metrics_acc["pixel_ious"].append(self._calculate_pixel_iou(y_pred_seg, y_true_seg))
                        segmentation_metrics_acc["tree_ious"].append(self._calculate_tree_iou(y_pred_seg, y_true_seg))
                        segmentation_metrics_acc["mean_ious"].append(self._calculate_mean_iou(y_pred_seg, y_true_seg))
                        segmentation_metrics_acc["balanced_ious"].append(self._calculate_balanced_iou(y_pred_seg, y_true_seg))
                        segmentation_metrics_acc["dice_scores"].append(self._calculate_dice_coefficient(y_pred_seg, y_true_seg))
                        segmentation_metrics_acc["adjusted_dice_scores"].append(
                            self._calculate_adjusted_dice_coefficient(y_pred_seg, y_true_seg)
                        )
                        segmentation_metrics_acc["mcc_scores"].append(self._calculate_mcc(y_pred_seg, y_true_seg))

                    for i in range(len(pred_centroids)):
                        y_pred_cent = pred_centroids[i].cpu().numpy()
                        y_true_cent = true_centroids[i].cpu().numpy()

                        pred_coords = self._extract_centroids_from_heatmap(y_pred_cent)
                        true_coords = self._extract_centroids_from_heatmap(y_true_cent)

                        precision, recall, f1_score = self._calculate_centroid_metrics(pred_coords, true_coords)
                        centroid_metrics_acc["precision"].append(precision)
                        centroid_metrics_acc["recall"].append(recall)
                        centroid_metrics_acc["f1_score"].append(f1_score)

                    pbar.update(images.size(0))
                    pbar.set_postfix(iterations_left=self.num_samples - pbar.n)

        segmentation_metrics = {key: np.mean(values) for key, values in segmentation_metrics_acc.items()}
        centroid_metrics = {key: np.mean(values) for key, values in centroid_metrics_acc.items()}

        return segmentation_metrics, centroid_metrics

    def _get_predictions(self, images):
        outputs = self.model(images)
        if self.model_name in ["maskformer", "detr"]:
            predictions = self._process_transformer_model(outputs, images.shape)
        elif self.model_name in ["dinov2", "beit"]:
            predictions = self._process_vision_transformer(outputs)
        else:
            predictions = torch.sigmoid(outputs[:, 0:1, :, :]), torch.sigmoid(outputs[:, 1:2, :, :])
        return predictions

    def _process_transformer_model(self, outputs, image_shape):
        query_logits = outputs["masks_queries_logits"] if "masks_queries_logits" in outputs else outputs["pred_masks"]
        combined_logits = torch.max(query_logits, dim=1).values
        h, w = image_shape[2], image_shape[3]
        interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(interpolated_logits), None

    def _process_vision_transformer(self, outputs):
        return torch.sigmoid(outputs.logits[:, 1:2, :, :]), None

    def _calculate_pixel_iou(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        if np.sum(y_pred_binary) == 0 and np.sum(y_true_binary) == 0:
            return 0.0

        intersection = (y_pred_binary * y_true_binary).sum()
        union = y_pred_binary.sum() + y_true_binary.sum() - intersection

        return (intersection + 1e-6) / (union + 1e-6)

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
            iou_trees = 1.0 if num_features_true == 0 else 0.0
        else:
            iou_trees = tp_trees / (tp_trees + fp_trees + fn_trees)

        return iou_trees

    def _calculate_mean_iou(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_fg = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_fg = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_fg = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))
        denominator_fg = tp_fg + fp_fg + fn_fg

        if denominator_fg == 0:
            iou_fg = 1.0 if tp_fg == 0 and (fp_fg == 0 and fn_fg == 0) else 0.0
        else:
            iou_fg = tp_fg / denominator_fg

        tp_bg = np.sum(np.logical_and(np.logical_not(y_pred_binary), np.logical_not(y_true_binary)))
        fp_bg = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))
        fn_bg = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        denominator_bg = tp_bg + fp_bg + fn_bg

        if denominator_bg == 0:
            iou_bg = 1.0 if tp_bg == 0 and (fp_bg == 0 and fn_bg == 0) else 0.0
        else:
            iou_bg = tp_bg / denominator_bg

        mean_iou = (iou_fg + iou_bg) / 2
        return mean_iou

    def _calculate_balanced_iou(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_fg = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_fg = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_fg = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))
        iou_fg = tp_fg / (tp_fg + fp_fg + fn_fg + 1e-10)

        tp_bg = np.sum(np.logical_and(np.logical_not(y_pred_binary), np.logical_not(y_true_binary)))
        fp_bg = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))
        fn_bg = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        iou_bg = tp_bg / (tp_bg + fp_bg + fn_bg + 1e-10)

        fg_weight = (np.sum(y_true_binary) / y_true_binary.size)
        bg_weight = 1.0 - fg_weight

        balanced_iou = fg_weight * iou_fg + bg_weight * iou_bg
        return balanced_iou

    def _calculate_dice_coefficient(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_pixels = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_pixels = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_pixels = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))

        denominator = 2 * tp_pixels + fp_pixels + fn_pixels

        dice = (2 * tp_pixels) / denominator if denominator != 0 else 0.0

        return dice

    def _calculate_adjusted_dice_coefficient(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_pixels = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_pixels = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_pixels = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))

        denominator = 2 * tp_pixels + fp_pixels + fn_pixels

        if denominator == 0:
            return 1.0 if np.array_equal(y_pred_binary, y_true_binary) else 0.0
        else:
            dice = (2 * tp_pixels) / denominator

        return dice

    def _calculate_mcc(self, y_pred, y_true):
        y_pred_binary = np.squeeze(y_pred > self.threshold)
        y_true_binary = np.squeeze(y_true > self.threshold)

        tp_pixels = np.sum(np.logical_and(y_pred_binary, y_true_binary))
        fp_pixels = np.sum(np.logical_and(y_pred_binary, np.logical_not(y_true_binary)))
        fn_pixels = np.sum(np.logical_and(np.logical_not(y_pred_binary), y_true_binary))
        tn_pixels = np.sum(
            np.logical_and(np.logical_not(y_pred_binary), np.logical_not(y_true_binary))
        )

        numerator = (tp_pixels * tn_pixels) - (fp_pixels * fn_pixels)
        denominator = np.sqrt(
            (tp_pixels + fp_pixels)
            * (tp_pixels + fn_pixels)
            * (tn_pixels + fp_pixels)
            * (tn_pixels + fn_pixels)
            + 1e-10
        )

        mcc = numerator / denominator
        return mcc

    def _extract_centroids_from_heatmap(self, heatmap, threshold=0.1, min_distance=5):
        if heatmap.ndim > 2:
            heatmap = np.squeeze(heatmap)

        if heatmap.ndim != 2:
            raise ValueError(f"Expected a 2D heatmap, got {heatmap.ndim} dimensions instead.")

        peaks = (heatmap == maximum_filter(heatmap, size=min_distance)) & (heatmap > threshold)

        return [tuple(coord) for coord in np.argwhere(peaks)]

    def _calculate_centroid_metrics(self, pred_coords, true_coords, proximity_threshold=5):
        if not pred_coords and not true_coords:
            return 1.0, 1.0, 1.0
        if not pred_coords or not true_coords:
            return 0.0, 0.0, 0.0

        distances = cdist(pred_coords, true_coords)
        matches = distances <= proximity_threshold

        tp = np.sum(np.any(matches, axis=1))
        fp = len(pred_coords) - tp
        fn = len(true_coords) - np.sum(np.any(matches, axis=0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1_score
