import os
import sys
from datetime import datetime
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, Lambda, MaxPool2D, Reshape, BatchNormalization

import pandas as pd
from data.datapipe import Datapipe
from losses import centerNetLoss
#from backends.efficientNetV2B0 import efficientNet
#from backends.efficientNetV2B0_v2 import efficientNet
from backends.efficientNetV3 import build_efficientnet_multihead
from callbacks import DrawImageCallback

# ========= Settings =================
ih,iw,ic = 256, 256, 3
ny,nx,nc = ih//4,iw//4,1



csvFilesTrain = [
    "thermalDet_train.csv",


]
csvFilesTest = [
    "thermalDet_test.csv",
]

NTRAIN = 245
NTEST = 62


learnrate = 1e-4
batchSize = 6

start_channels = 256
groups = 4

nfeatSN = 256
nfeat = 64


# ========= Datapipe =================

pipe = Datapipe()


g = pipe(csvFilesTrain, nx,ny,nc,iw,ih,ic, batchSize=batchSize)
gt  = pipe(csvFilesTest, nx,ny,nc,iw,ih,ic, augment=False, batchSize=batchSize)



# ========= Final prediction =================

model = build_efficientnet_multihead(input_shape=(ih,iw,ic), nfeat=64, nc=nc)

print(model.summary(line_length = 100))



# ============================================
# Training
# ============================================
now = datetime.now()
timestamp = str(now)[:19].replace(' ','_').replace(':','').replace('-','')
print(timestamp)

tfbcb = tf.keras.callbacks.TensorBoard(
    log_dir=f"./tblogs/efficientnet/{timestamp}", histogram_freq=0, write_graph=True,
    write_images=True, update_freq='batch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)

estcb = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0, patience=1400, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

mcpcb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(f'weights_efficientnet_{timestamp}.weights.h5'), monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
)

rlrcb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=100,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

dricb = DrawImageCallback(logdir=f"./tblogs/efficientnet/{timestamp}",tfdataset=gt, writerName="imagerVal",)
drtcb = DrawImageCallback(logdir=f"./tblogs/efficientnet/{timestamp}",tfdataset=g, writerName="imagerTrain",)
term =  tf.keras.callbacks.TerminateOnNaN()


#opti = tf.keras.optimizers.RMSprop(learning_rate=0.0006, clipnorm=5)
opti = tf.keras.optimizers.Adam(learnrate)


model.compile(
    loss=centerNetLoss,
    optimizer=opti
)

for x, y in g.take(1):
    print("Input shape:", x.shape)   # Should be (32, 224, 224, 3)
    print(x)
    print("Label shape:", y.shape)   # Should be (32, 10)
    print(y)
    ypred = model.predict(x)
    print("Pred shape:", ypred.shape)   # Should be (32, 10)




model.fit(
    g, epochs=3000,
   # callbacks = [tfbcb, mcpcb, estcb, rlrcb, dricb, drtcb, term],
    validation_data=gt,
    steps_per_epoch=NTRAIN//batchSize,
    validation_steps=NTEST//batchSize,
)



model.save_weights(f'weights_efficientnet_{timestamp}.h5')




