import tensorflow as tf


# Perus konvoluutio - bn - aktivaatio blokki
def cbr(x, filters, size=1, strides=1):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)  # relu6 aktivaatio
    return x


# Depthwise konvoluutio blokki
def dbr(x, size=3, strides=1):
    x = tf.keras.layers.DepthwiseConv2D(size, strides=strides, padding="same")(
        x
    )  # ei depth multiplyeria käytössä
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    return x


# mobile backbone inverted residual block - vähän modattuna
def mobile_res_block(input, filters, expansion, name, strides=1):
    input_channels = input.shape[-1]
    x = cbr(input, filters * expansion)
    x = dbr(x, strides=strides)
    x = tf.keras.layers.Conv2D(filters, 1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)  # linear output
    if (input_channels == filters) and (strides == 1):
        output = tf.keras.layers.Add(name=name)([input, x])
    else:
        output = x
    return output


# From tensorflow pix2pix example
def up_cbr(input_tensor, n_filters, name, kernel_size=3):
    x = tf.keras.layers.Conv2DTranspose(
        filters=n_filters,
        use_bias=False,
        strides=2,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
        name=(name + "_conv2D_trans"),
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization(name=(name + "_conv2dT_BN"))(x)
    x = tf.keras.layers.Activation("relu", name=(name + "_conv2dT_RELU"))(x)
    return x


# Define the Atrous Spatial Pyramid Pooling (ASPP) module
def ASPP(inputs, num_filters, atrous_rates):
    x = inputs
    branches = [tf.keras.layers.Conv2D(num_filters, 1, padding="same")(x)]
    for rate in atrous_rates:
        x_atrous = tf.keras.layers.DepthwiseConv2D(
            3, padding="same", dilation_rate=rate, use_bias=False
        )(x)
        x_atrous = tf.keras.layers.BatchNormalization()(x_atrous)
        x_atrous = tf.keras.layers.Activation("relu")(x_atrous)
        x_atrous = tf.keras.layers.Conv2D(num_filters, 1, padding="same")(x_atrous)
        branches.append(x_atrous)
    x = tf.keras.layers.Concatenate()(branches)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def Kokonet(input_shape, output_channels, activation):
    input = tf.keras.layers.Input(shape=input_shape)
    x_1 = cbr(input, 32, size=1, strides=1)

    # to 1/2 size
    x_2 = mobile_res_block(
        x_1, filters=128, expansion=1, name="bottleneck_2", strides=2
    )
    # # to 1/4 size
    x_4 = mobile_res_block(
        x_2, filters=256, expansion=6, name="bottleneck_4_a", strides=2
    )
    x_4 = mobile_res_block(
        x_4, filters=256, expansion=6, name="bottleneck_4_output", strides=1
    )

    aspp = ASPP(x_4, 256, atrous_rates=[2, 4, 8])  # dilated convolutions 2,3,4

    bottleneck_4 = cbr(aspp, 256, size=1, strides=1)

    # # upsample
    concat = tf.keras.layers.Concatenate()
    u_2 = up_cbr(bottleneck_4, n_filters=128, name="up_cbr_2", kernel_size=3)
    u_2 = concat([u_2, x_2])
    u_2 = cbr(u_2, 64, size=1, strides=1)
    u_1 = up_cbr(u_2, n_filters=32, name="up_cbr_1", kernel_size=3)
    u_1 = cbr(u_1, 32, size=1, strides=1)

    fex_out = concat([u_1, x_1])
    output = tf.keras.layers.Conv2D(
        output_channels,
        (1, 1),
        padding="same",
        name="segmentationmap",
        activation=activation,
    )(
        fex_out
    )  # activation from -1...1
    return tf.keras.Model(inputs=input, outputs=output)
