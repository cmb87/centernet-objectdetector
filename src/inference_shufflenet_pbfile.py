import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from postprocessing.freezer import ModelFreezer
import tensorflow as tf

def center_crop_square(frame):
    """
    Crops the frame to a centered square (removes sides if width > height).
    """
    h, w, _ = frame.shape
    if w == h:
        return frame  # already square

    if w > h:
        # Wider: crop left and right equally
        offset = (w - h) // 2
        cropped = frame[:, offset:offset + h]
    else:
        # Taller: crop top and bottom equally (rare for videos)
        offset = (h - w) // 2
        cropped = frame[offset:offset + w, :]

    return cropped


pbfile = "./models/frozen.pb"

iw,ih,nc = 2*128,2*128,1

colors = [(int(c[0]),int(c[1]),int(c[2])) for c in 255*cm.bwr(np.linspace(0, 1, nc))]


frozen = ModelFreezer.loadFrozen(
        pbfile,
        inputs=["x:0"],
        outputs=["Identity:0", "Identity_1:0", "Identity_2:0", "Identity_3:0"],
        print_graph=False,
)

cap = cv2.VideoCapture("/home/cp/projects/01_uxv/07-opticalTracker/opticaltracker/data/no_yolo_2025-07-16_13-43-10.avi")

# Get input video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, fps, (iw, ih))


while True:
    #frameOrg = cv2.imread("/SHARE4ALL/demoData/synthetic/000024.jpg")
    ret, frameOrg = cap.read()
   
    frameOrg = center_crop_square(frameOrg)
    frameOrg = cv2.resize(frameOrg, (iw,ih))
    frameOrg = cv2.bilateralFilter(frameOrg, d=5, sigmaColor=75, sigmaSpace=75)
    
    frame = cv2.cvtColor(frameOrg.copy(), cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame,0)
    frame = frame.astype(np.float32)
    frame = frame/255.0



    score,classes,bc,wh = frozen(tf.convert_to_tensor(frame))


    score,classes,bc,wh = score.numpy(),classes.numpy(),bc.numpy(),wh.numpy()

    # print(score.shape)
    # print(classes.shape)
    # print(bc.shape)
    # print(wh.shape)

    b = 0

    for s,c,b,w in zip(score[b],classes[b],bc[b],wh[b]):

        if float(s) > 0.1:
            frameOrg = cv2.rectangle(frameOrg, 
                (int(b[0]-0.5*w[0]), int(b[1]-0.5*w[1])), 
                (int(b[0]+0.5*w[0]), int(b[1]+0.5*w[1])),
                color=colors[int(c)],
                thickness=1
            )

            frameOrg = cv2.putText(frameOrg, f'{s*100:.2f}%', (int(b[0]-0.5*w[0]), int(b[1]-0.5*w[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(c)], 1, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame', frameOrg)
    out.write(frameOrg)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#s    cv2.imwrite("./models/inference.png", frameOrg)
  
# After the loop release the cap object
cap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()