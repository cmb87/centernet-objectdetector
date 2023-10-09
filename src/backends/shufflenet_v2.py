import os
import tensorflow as tf
import logging
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Multiply, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Lambda, MaxPool2D, Input, Conv2DTranspose, SeparableConv2D, ReLU, Permute, Reshape, DepthwiseConv2D,AvgPool2D,Concatenate,GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

from .layers import NormalizationLayer
from .layers import ByteLayer, ChannelAttentionLayer, SpatialAttentionLayer, WeightedAddLayer


def spatial_attention(x):
    # SA module
    nf = 256
    y = Conv2D(nf, kernel_size=1, padding = 'same')(x)
    y =  BatchNormalization()(y)
    y = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x) )(y)
    x = Multiply()([x,y])

    return x



def squeeze_excite_block(x, ratio=16):
    # store the input
    shortcut = x
    # calculate the number of filters the input has
    filters = x.shape[-1]
    # the squeeze operation reduces the input dimensionality
    # here we do a global average pooling across the filters, which
    # reduces the input to a 1D vector
    x = GlobalAveragePooling2D(keepdims=True)(x)
    # reduce the number of filters (1 x 1 x C/r)
    x = Dense(filters // ratio, activation="relu",
        kernel_initializer="he_normal", use_bias=False)(x)
    
    # the excitation operation restores the input dimensionality
    x = Dense(filters, activation="sigmoid",
        kernel_initializer="he_normal", use_bias=False)(x)
    
    # multiply the attention weights with the original input
    x = Multiply()([shortcut, x])
    # return the output of the SE block
    return x


def upsample(x, kernelsize=2, interpolation="bilinear"):
    return UpSampling2D(kernelsize, interpolation=interpolation)(x)


def convolution(x, filters, k, s=1, groups=None, bn=True, relu=True):
    if groups is None:
        x =  Conv2D (filters,kernel_size=k,strides = (s,s), padding = 'same', use_bias = True)(x)
    else:
        x =  Conv2D (filters,kernel_size=k,strides = (s,s), padding = 'same', use_bias = True, groups=groups)(x)
    x =  BatchNormalization()(x) if bn else x
    x =  ReLU()(x) if relu else x
    return x



def fire_module(x, fs, fe):
    s1 = convolution(x,  fs, k=1, s=1, groups=None)
    e1 = convolution(s1, fe, k=1, s=1, groups=None, bn=False, relu=False)
    e3 = convolution(s1, fe, k=3, s=1, groups=None, bn=False, relu=False)
    x = Add()([e1,e3])
    x =  BatchNormalization()(x)
    x =  ReLU()(x)
    return x
    


def channel_shuffle(x, groups):
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups
    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    return x


def deconv_layer(x, nf, k=3):
    x = Conv2DTranspose(nf, k, strides=(2,2), padding="same")(x)
    x =  BatchNormalization()(x)
    x =  ReLU()(x)
    return x


# ========================================
def shuffle_unit(x, groups, channels,strides, dilation_rate=(1, 1)):
    y = x

    x= convolution(x, filters=channels//4, k=1, s=1, groups=groups)

    x = channel_shuffle(x, groups)
    x = DepthwiseConv2D(kernel_size = (3,3), strides = strides, padding = 'same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    if strides == (2,2):
       channels = channels - y.shape[-1]

    x = Conv2D(channels, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
    x = BatchNormalization()(x)
    
    if strides ==(1,1):
        x = Add()([x,y])
    if strides == (2,2):  
        y = AvgPool2D((3,3), strides = (2,2), padding = 'same')(y)
        x = Concatenate()([x,y])
    x = ReLU()(x)
    return x


def Shuffle_Net(start_channels, groups = 2,  nf=256, input_shape = (224,224,3)):
    
    input = Input (input_shape)
    x = NormalizationLayer()(input)

    x = convolution(x, filters=24, k=3, s=2) # x2
    x4 = MaxPool2D (pool_size=(3,3), strides = 2, padding='same')(x) #x4

    # ============================================
    # Extractor
    # ============================================
    # x8 spatial reduction
    # Stage 2
    channels = start_channels * 1
    x = shuffle_unit(x4, groups, channels,strides = (2,2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(1, 1))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(2, 2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(1, 1))
    x8 = x

    # x16 spatial reduction
    # Stage 3
    channels = start_channels * 2
    x = shuffle_unit(x8, groups, channels,strides = (2,2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(1, 1))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(2, 2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(3, 3))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(1, 1))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(2, 2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(3, 3))
    x16 = x

    # x32, x4  spatial reduction, features
    # Stage 4
    channels = start_channels * 4
    x = shuffle_unit(x16, groups, channels,strides = (2,2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(1, 1))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(2, 2))
    x = shuffle_unit(x, groups, channels,strides=(1,1), dilation_rate=(3, 3))
    x32 = x

    # ============================================
    # Fusing
    # ============================================
    if True:


        upsample1 = Dropout(rate=0.500)(x32)
        upsample1 = ChannelAttentionLayer()(upsample1)
        upsample1 = SpatialAttentionLayer()(upsample1)
        upsample1 = upsample(upsample1, interpolation="bilinear")
        upsample1 = fire_module(upsample1, nf//2, nf)


        x = Dropout(rate=0.25)(x16)
        x = ChannelAttentionLayer()(x)
        x = SpatialAttentionLayer()(x)
        x = convolution(x,filters=nf,k=1,s=1)
        x = Add()([upsample1, x])


        upsample2 = ChannelAttentionLayer()(x)
        upsample2 = SpatialAttentionLayer()(upsample2)
        upsample2 = upsample(upsample2, interpolation="bilinear")
        upsample2 = fire_module(upsample2, nf//2, nf)


        x = Dropout(rate=0.25)(x8)
        x = ChannelAttentionLayer()(x)
        x = SpatialAttentionLayer()(x)
        x = convolution(x,filters=nf,k=1,s=1)

      #  x = WeightedAddLayer()([upsample2, x])
        x =Add()([upsample2, x])

        upsample3 = ChannelAttentionLayer()(x)
        upsample3 = SpatialAttentionLayer()(upsample3)
        upsample3 = upsample(upsample3, interpolation="bilinear")
        #upsample3 = convolution(upsample3,filters=nf,k=3,s=1)
        upsample3 = fire_module(upsample3, nf//2, nf)

        x = Dropout(rate=0.25)(x4)
        x = ChannelAttentionLayer()(x4)
        x = SpatialAttentionLayer()(x)
        x = convolution(x,filters=nf,k=1,s=1)

        #x = WeightedAddLayer()([upsample3, x])
        x = Add()([upsample3, x])
      #  x = upsample3

    # ============================================
    # Attention
    # ============================================
    y = x
    #y = squeeze_excite_block(x, ratio=16)
   # y = spatial_attention(x)




    model = Model(input, y)
    return model

if __name__ == "__main__":

    model = Shuffle_Net(start_channels=132, groups=3 ,input_shape = (224,224,3))



    print(model.summary())