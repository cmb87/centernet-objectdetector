import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import onnxruntime as ort




sess = ort.InferenceSession("./models/shufflenet_20230311_070949_pestControl.onnx")

# get model input details and resize image
input_details  = sess.get_inputs()[0]
output_details = sess.get_outputs()[0]

iw = input_details.shape[2]
ih = input_details.shape[1]
ic = input_details.shape[3]

nc = 4
colors = [(int(c[0]),int(c[1]),int(c[2])) for c in 255*cm.bwr(np.linspace(0, 1, nc))]
print(f"Model initialized {iw}, {ih}, {ic}")



if True:

    frameOrg = cv2.imread(r"C:\Users\z004hkut\projects\01_robotics\01_ml\project-ctchr\testimages\test3.png")

    frameOrg = cv2.resize(frameOrg, (iw,ih))
    frame = cv2.cvtColor(frameOrg.copy(), cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame,0)
    frame = frame.astype(np.float32)
    frame = frame/255.0



    score,classes,bc,wh = sess.run(None, {'x:0': frame})


    #score,classes,bc,wh = score[0],classes[0],bc[0],wh[0]

    print(score.shape)
    print(classes.shape)
    print(bc.shape)
    print(wh.shape)

    b = 0

    for s,c,b,w in zip(score[b],classes[b],bc[b],wh[b]):

        if float(s) > 0.15:
            frameOrg = cv2.rectangle(frameOrg, 
                (int(b[0]-0.5*w[0]), int(b[1]-0.5*w[1])), 
                (int(b[0]+0.5*w[0]), int(b[1]+0.5*w[1])),
                color=colors[int(c)],
                thickness=3
            )
    
    # Display the resulting frame
    cv2.imshow('frame', frameOrg)
    cv2.waitKey(0)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

#s    cv2.imwrite("./models/inference.png", frameOrg)
  
# After the loop release the cap object
# Destroy all the windows
cv2.destroyAllWindows()