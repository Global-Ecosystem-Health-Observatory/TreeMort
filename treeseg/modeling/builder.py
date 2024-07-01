import tensorflow as tf

from treeseg.modeling.network.kokonet import Kokonet


def build_model(model_name, input_channels, output_channels):
    assert model_name in ["unet", "kokonet"]

    if model_name == "unet":
        pass

    elif model_name == "kokonet":
        model = Kokonet(
            input_shape=[None, None, input_channels],
            output_channels=output_channels,
            activation="tanh",
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    model.compile(optimizer=optimizer, loss="mse")

    return model
