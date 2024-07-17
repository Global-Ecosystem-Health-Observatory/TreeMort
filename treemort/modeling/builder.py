import os

import tensorflow as tf
import segmentation_models as sm

from treemort.modeling.network.kokonet import Kokonet
from treemort.modeling.network.kokonet_hrnet import Kokonet_hrnet
from treemort.utils.checkpoints import get_checkpoint


def resume_or_load(conf):
    model = build_model(conf.model, conf.input_channels, conf.output_channels, conf.activation, conf.loss, conf.learning_rate, conf.threshold)

    if conf.resume:
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_weights(checkpoint, skip_mismatch=True)
            print(f"Loaded weights from {checkpoint}")

        else:
            print("No checkpoint found. Proceeding without loading weights.")

    else:
        print("Model training from scratch.")

    return model


def build_model(model_name, input_channels, output_channels, activation, loss, learning_rate, threshold):
    assert model_name in ["unet", "kokonet", "kokonet_hrnet"], f"Model {model_name} unavailable."
    assert activation in ["tanh", "sigmoid"], f"Model activation {activation} unavailable."
    assert loss in ["mse", "hybrid"], f"Model loss {loss} unavailable."

    if model_name == "unet":
        pass

    elif model_name == "kokonet":
        model = Kokonet(
            input_shape=[None, None, input_channels],
            output_channels=output_channels,
            activation=activation,
        )
    
    elif model_name == "kokonet_hrnet":
        model = Kokonet_hrnet(
            input_shape=[None, None, input_channels],
            output_channels=output_channels,
            activation=activation,
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if loss == "hybrid":
        iou_score = sm.metrics.IOUScore(threshold=threshold)
        f_score = sm.metrics.FScore(threshold=threshold)
        hybrid_metrics = [iou_score, f_score]

        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        hybrid_loss = dice_loss + (1 * focal_loss)

        model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=[hybrid_metrics])

    elif loss == "mse":
        mse_metric = tf.keras.metrics.MeanSquaredError(name='mse')
        mae_metric = tf.keras.metrics.MeanAbsoluteError(name='mae')
        rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='rmse')

        model.compile(optimizer=optimizer, loss='mse', metrics=[mse_metric, mae_metric, rmse_metric])

    return model
