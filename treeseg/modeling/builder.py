import tensorflow as tf
import segmentation_models as sm

from treeseg.modeling.network.kokonet import Kokonet
from treeseg.utils.checkpoints import get_checkpoint
from treeseg.utils.loss import focal_loss


def resume_or_load(conf, resume=True):
    model = build_model(conf.model, conf.input_channels, conf.output_channels, conf.activation, conf.learning_rate, conf.loss)

    if resume:
        checkpoint = get_checkpoint(conf.model_weights, conf.output_dir)

        if checkpoint:
            model.load_weights(checkpoint, skip_mismatch=True)
            print(f"Loaded weights from {checkpoint}")

        else:
            print("No checkpoint found. Proceeding without loading weights.")

    else:
        print("Model training from scratch.")

    return model


def build_model(model_name, input_channels, output_channels, activation, learning_rate, loss):
    assert model_name in ["unet", "kokonet"]
    assert loss in ["focal", "bce", "mse"]

    if model_name == "unet":
        pass

    elif model_name == "kokonet":
        model = Kokonet(
            input_shape=[None, None, input_channels],
            output_channels=output_channels,
            activation=activation,
        )

    '''
    if loss == "mse":
        loss_fn = tf.keras.losses.mean_squared_error()
    elif loss == "bce":
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    elif loss == "focal":
        #loss_fn = focal_loss(gamma=2., alpha=0.25)
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy()
    '''
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    iou_score = sm.metrics.IOUScore(threshold=0.5)
    f_score = sm.metrics.FScore(threshold=0.5)
    hybrid_metrics = [iou_score, f_score]

    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    hybrid_loss = dice_loss + (1 * focal_loss)

    model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=[hybrid_metrics])

    return model
