import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization


from backends.efficientNetV2B0 import efficientNet
from backends.layers import PostprocessingLayer
from postprocessing.freezer import ModelFreezer

# ========= Settings =================
ih,iw,ic = 128*4, 128*4, 3
ny,nx,nc = ih//4,iw//4, 1


start_channels = 256
groups = 4

nfeatSN = 256

nfeat = 256
nfeat = 64


# ========= Final prediction =================

model = efficientNet( input_size=512) # Shuffle_Net(start_channels=start_channels, groups=groups ,input_shape = (ih,iw,ic), nf=nfeatSN)

xhead1 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head1-conv13")(model.output)
xhead2 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head2-conv13")(model.output)
xhead3 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head3-conv13")(model.output)

xhead1 = Conv2D(nc, (1,1), padding="same", use_bias=True, activation="sigmoid", name="head1-conv21")(xhead1)
xhead2 = Conv2D(2, (1,1), padding="same", use_bias=True, name="head2-conv22")(xhead2)
xhead3 = Conv2D(2, (1,1), padding="same", use_bias=True, name="head3-conv23")(xhead3)

yhead = tf.keras.layers.Concatenate(axis=-1, name="head-final")([xhead1, xhead2, xhead3])

model = tf.keras.Model(inputs=model.inputs, outputs=yhead)

model.load_weights("./weights_efficientnet_20230716_074153.h5")
#model.load_weights("./models/weights_efficientnet_20230408_160754_pestControl.h5")
print(model.summary(line_length = 100))


# ========= Add postpressing head =================

ypproc = PostprocessingLayer(iw=iw, ih=ih, ndet=100, name="postpressing")(model.output)
modelPost = tf.keras.Model(inputs=model.inputs, outputs=ypproc)

print(modelPost.summary(line_length = 100))

# ========= Freeze model =================
ModelFreezer.convert2PbModel(modelPost, "./models")
ModelFreezer.convert2tflite(modelPost, "./models")
