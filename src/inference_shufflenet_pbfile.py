import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pbfile = "./shufflenet_20230311_070949_pestControl.onnx"


iw,ih,nc = 3*128,3*128,20

colors = [(int(c[0]),int(c[1]),int(c[2])) for c in 255*cm.bwr(np.linspace(0, 1, nc))]


frozen = ModelFreezer.loadFrozen(
        pbfile,
        inputs=["x:0"],
        outputs=["Identity:0", "Identity_1:0", "Identity_2:0", "Identity_3:0"],
        print_graph=False,
)

cap = cv2.VideoCapture(0)

while True:
    #frameOrg = cv2.imread("/SHARE4ALL/demoData/synthetic/000024.jpg")
    ret, frameOrg = cap.read()

    frameOrg = cv2.resize(frameOrg, (iw,ih))
    frame = cv2.cvtColor(frameOrg.copy(), cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame,0)
    frame = frame.astype(np.float32)
    frame = frame/255.0



    score,classes,bc,wh = frozen(tf.convert_to_tensor(frame))


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
    
    # Display the resulting frame
    cv2.imshow('frame', frameOrg)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#s    cv2.imwrite("./models/inference.png", frameOrg)
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()