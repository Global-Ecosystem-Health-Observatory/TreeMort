import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage


class IOUCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, dataset, num_samples, batch_size, threshold):
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
        pixel_ious = []
        tree_ious = []

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

                    y_pred_binary = tf.squeeze(y_pred[:, :, 0] > self.threshold)
                    y_true_binary = tf.squeeze(y_true > self.threshold)

                    # Pixel-wise IoU
                    tp_pixels = tf.reduce_sum(tf.logical_and(y_pred_binary, y_true_binary))
                    fp_pixels = tf.reduce_sum(tf.logical_and(y_pred_binary, tf.logical_not(y_true_binary)))
                    fn_pixels = tf.reduce_sum(tf.logical_and(tf.logical_not(y_pred_binary), y_true_binary))

                    if (tp_pixels + fp_pixels + fn_pixels) == 0:
                        if tf.reduce_sum(y_true_binary) == 0:
                            iou_pixels = 1.0  # All true negatives
                        else:
                            iou_pixels = 0.0  # Shouldn't happen, but handle gracefully
                    else:
                        iou_pixels = tp_pixels / (tp_pixels + fp_pixels + fn_pixels)
                    pixel_ious.append(iou_pixels)

                    # Tree-wise IoU
                    labeled_array_pred, num_features_pred = ndimage.label(y_pred_binary.numpy())
                    labeled_array_true, num_features_true = ndimage.label(y_true_binary.numpy())
                    predicted_contour_numbers = tf.unique(labeled_array_pred)
                    true_contour_numbers = tf.unique(labeled_array_true)

                    tp_trees, fp_trees, fn_trees = 0, 0, 0

                    for predicted_contour_number in predicted_contour_numbers:
                        if predicted_contour_number == 0:
                            continue
                        predicted_contour_mask = (labeled_array_pred == predicted_contour_number)
                        prediction_exists = tf.reduce_any(y_true_binary[predicted_contour_mask])
                        if prediction_exists:
                            tp_trees += 1
                        else:
                            fp_trees += 1

                    for true_contour_number in true_contour_numbers:
                        if true_contour_number == 0:
                            continue
                        true_contour_mask = labeled_array_true == true_contour_number
                        prediction_exists = tf.reduce_any(y_pred_binary[true_contour_mask])
                        if not prediction_exists:
                            fn_trees += 1

                    if (tp_trees + fp_trees + fn_trees) == 0:
                        if num_features_true == 0:
                            iou_trees = 1.0  # All true negatives
                        else:
                            iou_trees = 0.0  # Shouldn't happen, but handle gracefully
                    else:
                        iou_trees = tp_trees / (tp_trees + fp_trees + fn_trees)
                    tree_ious.append(iou_trees)

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(iterations_left=total_batches - pbar.n)

        mean_iou_pixels = tf.reduce_mean([iou for iou in pixel_ious if iou != -1])
        mean_iou_trees = tf.reduce_mean([iou for iou in tree_ious if iou != -1])

        return {"mean_iou_pixels": mean_iou_pixels, "mean_iou_trees": mean_iou_trees}
