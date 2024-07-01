import numpy as np
import tensorflow as tf

from tqdm import tqdm
from scipy import ndimage


class IOUCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, dataset, num_samples, batch_size, threshold=-0.5):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.threshold = threshold

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def evaluate(self):
        tp_pixels, fp_pixels, fn_pixels = 0, 0, 0
        tp_trees, fp_trees, fn_trees = 0, 0, 0

        total_batches = self.num_samples // self.batch_size

        with tqdm(total=total_batches, desc="Evaluating") as pbar:

            batch_iter = iter(self.dataset)

            for _ in range(total_batches):

                batch = next(batch_iter)
                images, labels = batch

                predictions = self.model.predict(images, verbose=0)

                for i in range(len(predictions)):
                    y_pred = predictions[i]
                    y_true = labels[i]

                    y_pred_binary = np.squeeze(y_pred[:, :, 0] > self.threshold)
                    y_true_binary = np.squeeze(y_true > self.threshold)

                    # Pixel-wise IoU
                    tp_pixels += np.sum(np.logical_and(y_pred_binary, y_true_binary))
                    fp_pixels += np.sum(
                        np.logical_and(y_pred_binary, np.logical_not(y_true_binary))
                    )
                    fn_pixels += np.sum(
                        np.logical_and(np.logical_not(y_pred_binary), y_true_binary)
                    )

                    # Tree-wise IoU
                    labeled_array_pred, num_features_pred = ndimage.label(y_pred_binary)
                    labeled_array_true, num_features_true = ndimage.label(y_true_binary)
                    predicted_contour_numbers = np.unique(labeled_array_pred)
                    true_contour_numbers = np.unique(labeled_array_true)

                    for predicted_contour_number in predicted_contour_numbers:
                        if predicted_contour_number == 0:
                            continue
                        predicted_contour_mask = (
                            labeled_array_pred == predicted_contour_number
                        )
                        prediction_exists = y_true_binary[predicted_contour_mask].any()
                        if prediction_exists:
                            tp_trees += 1
                        else:
                            fp_trees += 1

                    for true_contour_number in true_contour_numbers:
                        if true_contour_number == 0:
                            continue
                        true_contour_mask = labeled_array_true == true_contour_number
                        prediction_exists = y_pred_binary[true_contour_mask].any()
                        if not prediction_exists:
                            fn_trees += 1

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(iterations_left=total_batches - pbar.n)

        if (tp_pixels + fp_pixels + fn_pixels) == 0:
            iou_pixels = -1
        else:
            iou_pixels = tp_pixels / (tp_pixels + fp_pixels + fn_pixels)

        if (tp_trees + fp_trees + fn_trees) == 0:
            iou_trees = -1
        else:
            iou_trees = tp_trees / (tp_trees + fp_trees + fn_trees)

        return {"iou_pixels": iou_pixels, "iou_trees": iou_trees}
