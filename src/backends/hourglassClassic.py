import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization





class HourglassModule(tf.keras.Model):
    def __init__(self, nfilters, ndepths, useLight=False, useFire=False, **kwargs):
        super(HourglassModule, self).__init__(**kwargs)

        nf = nfilters[0]

        if useFire:
            self.low1 = FireModule(nf//2, nf, nf)
            self.low3 = FireModule(nf//2, nf, nf)
            self.up1 = FireModule(nf//2, nf, nf)
        else:
            self.low1 = Residual(nf)
            self.low3 = Residual(nf)
            self.up1 = Residual(nf)

       
        self.pool1 = Downsample(2)
        self.upsample = Upsample(2)

        if ndepths>1:
            self.hg = HourglassModule(nfilters=nfilters[1:], ndepths=ndepths-1)
        else:
            if useFire:
                self.hg = FireModule(nf//2, nf, nf)
            else:
                self.hg = Residual(nf)


    def call(self, x, training=None):

        up1  = self.up1(x, training=training)
        pool1 = self.pool1(x, training=training)
        low1 = self.low1(pool1, training=training)
        low2 = self.hg(low1, training=training)
        low3 = self.low3(low2, training=training)
        up2  = self.upsample(low3, training=training)

        return up1 + up2


