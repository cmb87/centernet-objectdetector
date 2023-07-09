import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization,ReLU
from keras.regularizers import l2

import pandas as pd
from data.datapipe import Datapipe
from losses import centerNetLoss
from backends.resnet import centernet
from backends.efficientNetV2B0 import efficientNet
from callbacks import DrawImageCallback

# ========= Settings =================
ih,iw,ic = 128*4, 128*4, 3
ny,nx,nc = ih//4,iw//4,4


csvFilesTrain = [
    "/SHARE4ALL/testData/stickytraps_train.csv",
    "/SHARE4ALL/testData/farmer1Tiny_train.csv",

]
csvFilesTest = [
    "/SHARE4ALL/testData/stickytraps_test.csv",
    "/SHARE4ALL/testData/farmer1Tiny_test.csv",
]

NTRAIN = 2720 + 81
NTEST = 303 + 10 


csvFilesTrain = [
    "/SHARE4ALL/testData/stickytrapsWOIn_train.csv",
    "/SHARE4ALL/testData/farmer1WOIn_train.csv",

]
csvFilesTest = [
    "/SHARE4ALL/testData/stickytrapsWOIn_test.csv",
    "/SHARE4ALL/testData/farmer1WOIn_test.csv",
]

NTRAIN = 2720 + 81
NTEST = 303 + 10 




learnrate = 1e-4
batchSize = 6



# ========= Datapipe =================

pipe = Datapipe()


g = pipe(csvFilesTrain, nx,ny,nc,iw,ih,ic, batchSize=batchSize)
gt  = pipe(csvFilesTest, nx,ny,nc,iw,ih,ic, augment=False, batchSize=batchSize)


# ========= Final prediction =================

model = centernet(num_classes=nc, backbone='resnet18', input_size=512, freeze_bn=True)
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

#model.load_weights("./weights_resnet_20230408_093200.h5")
model.load_weights("/SHARE4ALL/testData/weights_resnet_20230705_103701.h5")
print(model.summary(line_length = 100))


# ============================================
# Training
# ============================================
now = datetime.now()
timestamp = str(now)[:19].replace(' ','_').replace(':','').replace('-','')
print(timestamp)

tfbcb = tf.keras.callbacks.TensorBoard(
    log_dir=f"./tblogs/resnet/{timestamp}", histogram_freq=0, write_graph=True,
    write_images=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

estcb = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0, patience=1400, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

mcpcb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(f'weights_resnet_{timestamp}.h5'), monitor='loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
)

rlrcb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=35,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

dricb = DrawImageCallback(logdir=f"./tblogs/resnet/{timestamp}",tfdataset=gt, writerName="imagerVal",)
drtcb = DrawImageCallback(logdir=f"./tblogs/resnet/{timestamp}",tfdataset=g, writerName="imagerTrain",)
term =  tf.keras.callbacks.TerminateOnNaN()


#opti = tf.keras.optimizers.RMSprop(learning_rate=0.0006, clipnorm=5)
opti = tf.keras.optimizers.Adam(learnrate)


model.compile(
    loss=centerNetLoss,
    optimizer=opti
)


model.fit(
    g, epochs=3000,
    callbacks = [tfbcb, mcpcb, estcb, rlrcb, dricb, drtcb, term],
    validation_data=gt,
    steps_per_epoch=NTRAIN//batchSize,
    validation_steps=NTEST//batchSize,
)



model.save_weights(f'weights_resnet_{timestamp}.h5')




