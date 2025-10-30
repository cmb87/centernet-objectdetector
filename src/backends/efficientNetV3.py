import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras.models import Model

def build_efficientnet_multihead(input_shape=(256, 256, 3), nfeat=64, nc=1):
    """
    Build an EfficientNet backbone with three parallel Conv2D heads concatenated.
    
    Args:
        input_shape (tuple): input tensor shape (H, W, C)
        nfeat (int): number of feature channels for intermediate convs
        nc (int): number of output channels for the first head (e.g., class maps)
    
    Returns:
        tf.keras.Model: complete model ready for training
    """

    # --- Backbone ---
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(256,256,3),
        weights=None,
    )


    print(base_model.summary())
    x = base_model.output  # feature map from EfficientNet

    # Optional: add a feature aggregation layer (if you want smaller channels)
    # x = Conv2D(nfeat, (1, 1), activation="relu", padding="same", name="reduce-dim")(x)

    # --- Head 1 ---
    xhead1 = Conv2D(
        nfeat, (3, 3), padding="same", activation="relu", name="head1-conv13"
    )(x)
    xhead1 = Conv2D(
        nc, (1, 1), padding="same", activation="sigmoid", name="head1-conv21"
    )(xhead1)

    # --- Head 2 ---
    xhead2 = Conv2D(
        nfeat, (3, 3), padding="same", activation="relu", name="head2-conv13"
    )(x)
    xhead2 = Conv2D(
        2, (1, 1), padding="same", name="head2-conv22"
    )(xhead2)

    # --- Head 3 ---
    xhead3 = Conv2D(
        nfeat, (3, 3), padding="same", activation="relu", name="head3-conv13"
    )(x)
    xhead3 = Conv2D(
        2, (1, 1), padding="same", name="head3-conv23"
    )(xhead3)

    # --- Combine all heads ---
    yhead = Concatenate(axis=-1, name="head-final")([xhead1, xhead2, xhead3])

    # --- Define the final model ---
    model = Model(inputs=base_model.input, outputs=yhead, name="EfficientNet_MultiHead")

    return model
