from keras_resnet import models as resnet_models
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout, Add
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
from .layers import ByteLayer, ChannelAttentionLayer, SpatialAttentionLayer, WeightedAddLayer

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

    for n,l in enumerate(efficientNet.layers):
        print(n, l.name, l.get_output_at(0).get_shape().as_list())

    C2 = efficientNet.layers[18].output # 128
    C3 = efficientNet.layers[32].output # 64
    C4 = efficientNet.layers[142].output # 32
    C5 = efficientNet.outputs[-1] #16

    x2 = Dropout(rate=0.125)(C2)
    x3 = Dropout(rate=0.125)(C3)
    x4 = Dropout(rate=0.250)(C4)
    x5 = Dropout(rate=0.500)(C5)

    x = x5
    num_filters = 256

    for i,y in enumerate([x4,x3,x2]):
        num_filters = num_filters // pow(2, i)
        # x = Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(
        #     x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        #x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
        #                    kernel_initializer='he_normal',
        #                    kernel_regularizer=l2(5e-4))(x)
        
        x = ChannelAttentionLayer()(x)
        x = SpatialAttentionLayer()(x)
        x = Conv2D(num_filters, (1, 1), padding='same')(x)
        x  = tf.keras.layers.UpSampling2D( size=(2, 2), data_format=None, interpolation='bilinear')(x)

        y = ChannelAttentionLayer()(y)
        y = SpatialAttentionLayer()(y)
        y = Conv2D(num_filters, (1, 1), padding='same')(y)

        #x = Add()([x,y])
        x = WeightedAddLayer()([x,y])
        x = BatchNormalization()(x)
        x = ReLU()(x)



    model = Model(inputs=image_input, outputs=x)


    return model

if (__name__ == "__main__"):
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"


    model = efficientNet( input_size=512)
    print(model.summary())