import torch
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import cdist


class IOUCallback:
    def __init__(
        self,
        model,
        dataset,
        num_samples,
        batch_size,
        threshold,
        model_name,
    ):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_name = model_name
        self.device = next(model.parameters()).device

    def evaluate(self):
        self.model.eval()

        (pixel_ious, tree_ious, mean_ious, balanced_ious, dice_scores, adjusted_dice_scores, mcc_scores) = ([], [], [], [], [], [], [])
        (centroid_precisions, centroid_recalls, centroid_f1s) = ([], [], [])

        with tqdm(total=self.num_samples, desc="Evaluating") as pbar:
            with torch.no_grad():
                for images, labels in self.dataset:
                    images, labels = images.to(self.device), labels.to(self.device)

                    pred_segmentation, pred_centroids = self._get_predictions(images, labels)

                    true_segmentation = labels[:, 0, :, :]
                    true_centroids = labels[:, 1, :, :]

                    for i in range(len(pred_segmentation)):
                        y_pred_seg = pred_segmentation[i].cpu().numpy()
                        y_true_seg = true_segmentation[i].cpu().numpy()

                        pixel_ious.append(self._calculate_pixel_iou(y_pred_seg, y_true_seg))
                        tree_ious.append(self._calculate_tree_iou(y_pred_seg, y_true_seg))
                        mean_ious.append(self._calculate_mean_iou(y_pred_seg, y_true_seg))
                        balanced_ious.append(self._calculate_balanced_iou(y_pred_seg, y_true_seg))
                        dice_scores.append(self._calculate_dice_coefficient(y_pred_seg, y_true_seg))
                        adjusted_dice_scores.append(self._calculate_adjusted_dice_coefficient(y_pred_seg, y_true_seg))
                        mcc_scores.append(self._calculate_mcc(y_pred_seg, y_true_seg))

                    for i in range(len(pred_centroids)):
                        y_pred_cent = (pred_centroids[i] > self.threshold).cpu().numpy()
                        y_true_cent = (true_centroids[i] > self.threshold).cpu().numpy()

                        precision, recall, f1 = self._calculate_centroid_metrics(y_pred_cent, y_true_cent)
                        centroid_precisions.append(precision)
                        centroid_recalls.append(recall)
                        centroid_f1s.append(f1)

                    pbar.update(1)
                    pbar.set_postfix(iterations_left=self.num_samples - pbar.n)

        segmentation_metrics = self._compute_mean_ious(
            pixel_ious, tree_ious, mean_ious, balanced_ious, dice_scores, adjusted_dice_scores, mcc_scores
        )
        centroid_metrics = {
            "precision": np.mean(centroid_precisions),
            "recall": np.mean(centroid_recalls),
            "f1_score": np.mean(centroid_f1s),
        }

        return segmentation_metrics, centroid_metrics

    def _get_predictions(self, images, labels):
        outputs = self.model(images)

        _, _, h, w = images.shape

        if self.model_name == "maskformer":
            query_logits = outputs['masks_queries_logits']
            combined_logits = torch.max(query_logits, dim=1).values
            interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
            predictions = torch.sigmoid(interpolated_logits)

        elif self.model_name == "detr":
            query_logits = outputs['pred_masks']
            combined_logits = torch.max(query_logits, dim=1).values
            interpolated_logits = F.interpolate(combined_logits.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
            predictions = torch.sigmoid(interpolated_logits)

        elif self.model_name in ["dinov2", "beit"]:
            logits = outputs.logits[:, 1:2, :, :]
            predictions = torch.sigmoid(logits)

        else:
            segmentation_predictions = torch.sigmoid(outputs[:, 0:1, :, :])
            centroid_predictions = torch.sigmoid(outputs[:, 1:2, :, :])

        return segmentation_predictions, centroid_predictions

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

    def _compute_mean_ious(
        self,
        pixel_ious,
        tree_ious,
        mean_ious,
        balanced_ious,
        dice_scores,
        adjusted_dice_scores,
        mcc_scores,
    ):
        mean_iou_pixels = np.mean(pixel_ious)
        mean_iou_trees = np.mean(tree_ious)
        mean_iou = np.mean(mean_ious)
        mean_balanced_iou = np.mean(balanced_ious)
        mean_dice_score = np.mean(dice_scores)
        mean_adjusted_dice_score = np.mean(adjusted_dice_scores)
        mean_mcc = np.mean(mcc_scores)

        return {
            "mean_iou_pixels": mean_iou_pixels,
            "mean_iou_trees": mean_iou_trees,
            "mean_iou": mean_iou,
            "mean_balanced_iou": mean_balanced_iou,
            "mean_dice_score": mean_dice_score,
            "mean_adjusted_dice_score": mean_adjusted_dice_score,
            "mean_mcc": mean_mcc,
        }

    def _calculate_centroid_metrics(self, y_pred_centroids, y_true_centroids, proximity_threshold=5):
        pred_coords = np.argwhere(y_pred_centroids > 0)
        true_coords = np.argwhere(y_true_centroids > 0)

        if len(pred_coords) == 0 and len(true_coords) == 0:
            return 1.0, 1.0, 1.0  # Perfect match when there are no centroids

        if len(pred_coords) == 0 or len(true_coords) == 0:
            return 0.0, 0.0, 0.0  # No match if one set is empty

        distances = cdist(pred_coords, true_coords)

        matches = distances <= proximity_threshold

        tp = np.sum(np.any(matches, axis=1))

        fp = len(pred_coords) - tp

        fn = len(true_coords) - np.sum(np.any(matches, axis=0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1