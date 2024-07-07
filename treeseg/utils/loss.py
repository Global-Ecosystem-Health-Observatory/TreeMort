import tensorflow as tf


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Convert tanh output to sigmoid range [0, 1]
        y_pred = (y_pred + 1) / 2
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute binary cross-entropy
        bce = - (alpha * y_true * tf.math.log(y_pred) + (1 - alpha) * (1 - y_true) * tf.math.log(1 - y_pred))

        # Compute focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = tf.pow(1. - pt, gamma) * bce

        return tf.reduce_sum(fl)
    return focal_loss_fixed

