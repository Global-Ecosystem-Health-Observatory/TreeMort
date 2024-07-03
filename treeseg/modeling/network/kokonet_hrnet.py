import tensorflow as tf
from segmentation_models import Unet
from segmentation_models import get_preprocessing

# Define the HRNet backbone
BACKBONE = 'efficientnetb0'
preprocess_input = get_preprocessing(BACKBONE)

def Kokonet_hrnet(input_shape=(256, 256, 4), output_channels=1, activation="tanh"):
    # Input layer to match aerial imagery shape (256, 256, 4)
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Adjust input channels to 3 to match HRNet requirements
    adjusted_inputs = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(inputs)
    adjusted_inputs = preprocess_input(adjusted_inputs)

    # Create the HRNet backbone model
    hrnet_backbone = Unet(BACKBONE, input_shape=(256, 256, 3), encoder_weights='imagenet', classes=output_channels, activation=None)
    
    # Extract features from the HRNet backbone
    hrnet_features = hrnet_backbone.encoder(adjusted_inputs)
    
    # Example feature layers extraction
    x_1 = hrnet_features[0]  # Extract intermediate feature layer
    x_2 = hrnet_features[1]
    x_4 = hrnet_features[2]

    # Apply ASPP
    aspp = ASPP(x_4, 256, atrous_rates=[2, 4, 8])

    # Decoder: upsampling and concatenation
    concat = tf.keras.layers.Concatenate()
    u_2 = up_cbr(aspp, n_filters=128, name="up_cbr_2", kernel_size=3)
    u_2 = concat([u_2, x_2])
    u_2 = cbr(u_2, 64, size=1, strides=1)
    u_1 = up_cbr(u_2, n_filters=32, name="up_cbr_1", kernel_size=3)
    u_1 = concat([u_1, x_1])
    u_1 = cbr(u_1, 32, size=1, strides=1)
    
    # Final output layer
    output = tf.keras.layers.Conv2D(
        output_channels,
        (1, 1),
        padding="same",
        name="segmentationmap",
        activation=activation
    )(u_1)

    return tf.keras.Model(inputs=inputs, outputs=output)

def cbr(x, filters, size=1, strides=1):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    return x

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
