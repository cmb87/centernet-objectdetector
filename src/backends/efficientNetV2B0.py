from keras_resnet import models as resnet_models
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
from .layers import ByteLayer

# https://github.com/xuannianz/keras-CenterNet/blob/master/models/resnet.py

def efficientNet( input_size=512):

    image_input = Input(shape=(input_size, input_size, 3))
    x = ByteLayer()(image_input)

    preModel = Model(inputs=image_input, outputs=x)


    efficientNet = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=preModel.outputs[0],
        include_preprocessing=True
    )
 
    efficientNet.trainable =False

    C5 = efficientNet.outputs[-1]
    # C5 = resnet.get_layer('activation_49').output

    x = Dropout(rate=0.5)(C5)
    # decoder
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        # x = Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(
        #     x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)


    model = Model(inputs=image_input, outputs=x)


    return model

if (__name__ == "__main__"):

    model = efficientNet( input_size=512)
    print(model.summary())