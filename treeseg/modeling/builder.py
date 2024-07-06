import tensorflow as tf

from treeseg.modeling.network.kokonet import Kokonet
from treeseg.utils.checkpoints import get_checkpoint
from treeseg.utils.loss import focal_loss


def resume_or_load(conf, resume):
    model = build_model(conf.model, conf.input_channels, conf.output_channels, conf.learning_rate)

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


def build_model(model_name, input_channels, output_channels, learning_rate):
    assert model_name in ["unet", "kokonet"]

    if model_name == "unet":
        pass

    elif model_name == "kokonet":
        model = Kokonet(
            input_shape=[None, None, input_channels],
            output_channels=output_channels,
            activation="tanh",
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #model.compile(optimizer=optimizer, loss="mse")
    model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])

    return model
