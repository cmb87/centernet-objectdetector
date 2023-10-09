import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization

import pandas as pd
from data.datapipe import Datapipe
from losses import centerNetLoss
from callbacks import DrawImageCallback
from backends.layers import getHourglass

# ========= Settings =================
ih,iw,ic = 128*4, 128*4, 3
ny,nx,nc = ih//4,iw//4,20


csvFilesTrain = [
    "/SHARE4ALL/pascalVOC/VOC2007p12/VOC2007_train.csv",
  #  "/SHARE4ALL/pascalVOC/VOC2007p12/VOC2007p12_train.csv"
]
csvFilesTest = [
    "/SHARE4ALL/pascalVOC/VOC2007p12/VOC2007_test.csv",
   # "/SHARE4ALL/pascalVOC/VOC2007p12/VOC2007p12_test.csv"
]

#NTEST = 992+4429 -len(csvFilesTest)
#NTRAIN = 3962 + 17709 -len(csvFilesTrain)

NTEST = 992 -len(csvFilesTest)
NTRAIN = 3962 -len(csvFilesTrain)


learnrate = 1e-4
batchSize = 7

start_channels = 256
groups = 4

nfeatSN = 256

nfeat = 256

# ========= Datapipe =================

pipe = Datapipe()


g = pipe(csvFilesTrain, nx,ny,nc,iw,ih,ic, batchSize=batchSize)
gt  = pipe(csvFilesTest, nx,ny,nc,iw,ih,ic, augment=False, batchSize=batchSize)


# ========= Final prediction =================

model = getHourglass(512, 512, 3, nfeat=128, nfilters=[128,128,256,256,512])

xhead1 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head1-conv13")(model.output)
xhead2 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head2-conv13")(model.output)
xhead3 = Conv2D(nfeat, (3,3), padding="same", use_bias=True, activation="relu", name="head3-conv13")(model.output)

xhead1 = Conv2D(nc, (1,1), padding="same", use_bias=True, activation="sigmoid", name="head1-conv21")(xhead1)
xhead2 = Conv2D(2, (1,1), padding="same", use_bias=True, name="head2-conv22")(xhead2)
xhead3 = Conv2D(2, (1,1), padding="same", use_bias=True, name="head3-conv23")(xhead3)

yhead = tf.keras.layers.Concatenate(axis=-1, name="head-final")([xhead1, xhead2, xhead3])

model = tf.keras.Model(inputs=model.inputs, outputs=yhead)

#model.load_weights("weights_hourglass_20230724_181757.h5")

print(model.summary(line_length = 100))


# ============================================
# Training
# ============================================
now = datetime.now()
timestamp = str(now)[:19].replace(' ','_').replace(':','').replace('-','')
print(timestamp)

tfbcb = tf.keras.callbacks.TensorBoard(
    log_dir=f"./tblogs/hourglass/{timestamp}", histogram_freq=0, write_graph=True,
    write_images=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

estcb = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0, patience=1400, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

mcpcb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(f'weights_hourglass_{timestamp}.h5'), monitor='loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
)

rlrcb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=45,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

dricb = DrawImageCallback(logdir=f"./tblogs/hourglass/{timestamp}",tfdataset=gt, writerName="imagerVal",)
drtcb = DrawImageCallback(logdir=f"./tblogs/hourglass/{timestamp}",tfdataset=g, writerName="imagerTrain",)
term =  tf.keras.callbacks.TerminateOnNaN()


def scheduler(epoch, lr):
    if epoch < 120:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    

lrscb = tf.keras.callbacks.LearningRateScheduler(scheduler)
opti = tf.keras.optimizers.Adam(learnrate)


model.compile(
    loss=centerNetLoss,
    optimizer=opti
)


model.fit(
    g, epochs=3000,
    callbacks = [tfbcb, mcpcb, estcb, rlrcb, dricb, drtcb, term, lrscb],
    validation_data=gt,
    steps_per_epoch=NTRAIN//batchSize,
    validation_steps=NTEST//batchSize,
)



model.save_weights(f'weights_hourglass_{timestamp}.h5')




