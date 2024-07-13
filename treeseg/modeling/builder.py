import os

#import tensorflow as tf
#import segmentation_models as sm

from treeseg.modeling.network.self_attention_unet import SelfAttentionUNet
from treeseg.utils.checkpoints import get_checkpoint


def resume_or_load(conf):
    model = SelfAttentionUNet(conf.input_channels, conf.output_channels, depth=4, wf=6, batch_norm=True)
    #model = build_model(conf.model, conf.input_channels, conf.output_channels, conf.activation, conf.learning_rate, conf.threshold)

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


def build_model(model_name, input_channels, output_channels, activation, learning_rate, threshold):
    assert model_name in ["unet", "kokonet", "kokonet_hrnet"], f"Model {model_name} unavailable."
    assert activation in ["tanh", "sigmoid"], f"Model activation {activation} unavailable."

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

    iou_score = sm.metrics.IOUScore(threshold=threshold)
    f_score = sm.metrics.FScore(threshold=threshold)
    hybrid_metrics = [iou_score, f_score]

    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    hybrid_loss = dice_loss + (1 * focal_loss)

    model.compile(optimizer=optimizer, loss=hybrid_loss, metrics=[hybrid_metrics])

    return model
