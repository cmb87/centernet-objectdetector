import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization,ReLU
from keras.regularizers import l2

from backends.resnet import centernet
from backends.efficientNetV2B0 import efficientNet
from backends.layers import PostprocessingLayer
from postprocessing.freezer import ModelFreezer

# ========= Settings =================
ih,iw,ic = 128*3, 128*3, 3
ny,nx,nc = ih//4,iw//4, 4

start_channels = 256
groups = 4


# ========= Final prediction =================


model = centernet(num_classes=nc, backbone='resnet18', input_size=512, freeze_bn=False)
#model = efficientNet( input_size=512)
x = model.outputs[0]
# hm header
y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
y1 = BatchNormalization()(y1)
y1 = ReLU()(y1)
y1 = Conv2D(nc, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

# wh header
y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
y2 = BatchNormalization()(y2)
y2 = ReLU()(y2)
y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

# reg header
y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
y3 = BatchNormalization()(y3)
y3 = ReLU()(y3)
y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

yhead = tf.keras.layers.Concatenate(axis=-1, name="head-final")([y1, y2, y3])

model = tf.keras.Model(inputs=model.inputs, outputs=yhead)


model.load_weights("/SHARE4ALL/testData/weights_resnet_20230705_103701.h5")

print(model.summary(line_length = 100))


# ========= Add postpressing head =================

ypproc = PostprocessingLayer(iw=iw, ih=ih, ndet=300, name="postpressing")(model.output)
modelPost = tf.keras.Model(inputs=model.inputs, outputs=ypproc)

print(modelPost.summary(line_length = 100))

# ========= Freeze model =================
ModelFreezer.convert2PbModel(modelPost, "./models")
ModelFreezer.convert2tflite(modelPost, "./models")

