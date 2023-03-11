import os
import cv2
import sys
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

from postprocessing.freezer import ModelFreezer

pbfile = "./models/shufflenet_20230311_070949_pestControl.pb"



frozen = ModelFreezer.loadFrozen(
        pbfile,
        inputs=["x:0"],
        outputs=["Identity:0", "Identity_1:0", "Identity_2:0", "Identity_3:0"],
        print_graph=False,
)


frameOrg = cv2.imread("/SHARE4ALL/demoData/synthetic/000024.jpg")
frameOrg = cv2.resize(frameOrg, (4*128, 4*128))
frame = cv2.cvtColor(frameOrg.copy(), cv2.COLOR_BGR2RGB)
frame = np.expand_dims(frame,0)
frame = frame.astype(np.float32)
frame = frame/255.0



score,classes,bc,wh = frozen(tf.convert_to_tensor(frame))

colors = [(255,0,0), (0,255,0), (0,0,255) ,(0,255,255)]



score,classes,bc,wh = score.numpy(),classes.numpy(),bc.numpy(),wh.numpy()

print(score.shape)
print(classes.shape)
print(bc.shape)
print(wh.shape)

b = 0

for s,c,b,w in zip(score[b],classes[b],bc[b],wh[b]):

    if float(s) > 0.3:
        frameOrg = cv2.rectangle(frameOrg, 
            (int(b[0]-0.5*w[0]), int(b[1]-0.5*w[1])), 
            (int(b[0]+0.5*w[0]), int(b[1]+0.5*w[1])),
            color=colors[int(c)],
            thickness=3
        )
    

cv2.imwrite("./models/inference.png", frameOrg)
