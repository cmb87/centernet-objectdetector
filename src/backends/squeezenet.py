import os
import tensorflow as tf
import logging
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Lambda, MaxPool2D, Input, Conv2DTranspose, SeparableConv2D, ReLU
from tensorflow.keras.regularizers import l2




# ===========================================================
# The Feature extractor
# ===========================================================
def squeezenet(imageWidth, imageHeight, imageChannels=3, include_top=True, pretrained=True, lastTrainableLayers=0):
    # SQUEEZE NET V1.1 : A LEX N ET- LEVEL ACCURACY WITH
    # 50 X FEWER PARAMETERS AND <0.5MB MODEL SIZE
    inputs = Input((imageHeight, imageWidth, imageChannels))

    n1 = CafeNormalizationLayer(name="CafeNormalization")(inputs)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv1", strides=(2, 2))(n1)
    p1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1", padding='same')(c1)

    # C1

    f2s1 = Conv2D(16, (1, 1), activation='relu', padding='same', name="Fire2s1")(p1)
    f2e1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire2e1")(f2s1)
    f2e3 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Fire2e3")(f2s1)
    f2 = Concatenate(name="Fire2cat")([f2e1, f2e3])

    f3s1 = Conv2D(16, (1, 1), activation='relu', padding='same', name="Fire3s1")(f2)
    f3e1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire3e1")(f3s1)
    f3e3 = Conv2D(64, (3, 3), activation='relu', padding='same', name="Fire3e3")(f3s1)
    f3 = Concatenate(name="Fire3cat")([f3e1, f3e3])

    p3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool3", padding='same')(f3)

    # C2

    f4s1 = Conv2D(32, (1, 1), activation='relu', padding='same', name="Fire4s1")(p3)
    f4e1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="Fire4e1")(f4s1)
    f4e3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Fire4e3")(f4s1)
    f4 = Concatenate(name="Fire4cat")([f4e1, f4e3])

    f5s1 = Conv2D(32, (1, 1), activation='relu', padding='same', name="Fire5s1")(f4)
    f5e1 = Conv2D(128, (1, 1), activation='relu', padding='same', name="Fire5e1")(f5s1)
    f5e3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="Fire5e3")(f5s1)
    f5 = Concatenate(name="Fire5cat")([f5e1, f5e3])

    p5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool5", padding='same')(f5)

    # C3

    f6s1 = Conv2D(48, (1, 1), activation='relu', padding='same', name="Fire6s1")(p5)
    f6e1 = Conv2D(192, (1, 1), activation='relu', padding='same', name="Fire6e1")(f6s1)
    f6e3 = Conv2D(192, (3, 3), activation='relu', padding='same', name="Fire6e3")(f6s1)
    f6 = Concatenate(name="Fire6cat")([f6e1, f6e3])

    f7s1 = Conv2D(48, (1, 1), activation='relu', padding='same', name="Fire7s1")(f6)
    f7e1 = Conv2D(192, (1, 1), activation='relu', padding='same', name="Fire7e1")(f7s1)
    f7e3 = Conv2D(192, (3, 3), activation='relu', padding='same', name="Fire7e3")(f7s1)
    f7 = Concatenate(name="Fire7cat")([f7e1, f7e3])

    # C4

    f8s1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire8s1")(f7)
    f8e1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="Fire8e1")(f8s1)
    f8e3 = Conv2D(256, (3, 3), activation='relu', padding='same', name="Fire8e3")(f8s1)
    f8 = Concatenate(name="Fire8cat")([f8e1, f8e3])

    f9s1 = Conv2D(64, (1, 1), activation='relu', padding='same', name="Fire9s1")(f8)
    f9e1 = Conv2D(256, (1, 1), activation='relu', padding='same', name="Fire9e1")(f9s1)
    f9e3 = Conv2D(256, (3, 3), activation='relu', padding='same', name="Fire9e3")(f9s1)
    f9 = Concatenate(name="Fire9cat")([f9e1, f9e3])
    x = f9

    # C5

    FILEDIR = os.path.dirname(os.path.realpath(__file__))

    # Build basic modelWithoutPosthead
    model = Model(inputs=[inputs], outputs=[x], name='squeezenet_extractor')
    model.load_weights(os.path.join(FILEDIR,'squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5'))

    # Set layers to trainable=False, except for lastTrainableLayers
    for l in range(len(model.layers) - lastTrainableLayers):
        model.layers[l].trainable = False

    return model
