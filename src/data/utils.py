import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os, json, glob
from pascal import PascalVOC


# =================================================
def readJsonAnnotation(jsonfile, datapath, classNames, minLength=10, renameDict={}, **kwargs):
    """Read JSON Annotation LABELME"""
    with open(jsonfile, 'r') as f1:
        data = json.load(f1)

    imgpath = os.path.join(datapath, data["imagePath"])

    size = [data["imageWidth"], data["imageHeight"]]

    boxes, labels = [], []

    for obj in data['shapes']:
        if obj["shape_type"] == "rectangle":
            label = obj['label']

            if label in renameDict.keys():
                label = renameDict[label]

            if label not in classNames:
                continue

            x1 = min((obj['points'][0][0], obj['points'][1][0])) / data["imageWidth"]
            x2 = max((obj['points'][0][0], obj['points'][1][0])) / data["imageWidth"]
            y1 = min((obj['points'][0][1], obj['points'][1][1])) / data["imageHeight"]
            y2 = max((obj['points'][0][1], obj['points'][1][1])) / data["imageHeight"]

            # Dont consider when too small
            if np.sqrt(size[0]*size[1]*(x2-x1)*(y2-y1)) < minLength:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(classNames.index(label))

    if len(boxes) == 0:
        boxes = np.zeros((0, 4))


    return imgpath, boxes, labels, size


# =================================================
def readPASCALVOCAnnotation(xmlfile, datapath, classNames, minLength=10, renameDict={}, **kwargs):
    """PASCAL VOC"""

    ann = PascalVOC.from_xml(xmlfile)
    imgpath = os.path.join(datapath, ann.filename)
    w, h = ann.size.width, ann.size.height
        
    boxes, labels = [], []

    for obj in ann.objects:
        label = obj.name

        if label in renameDict.keys():
            label = renameDict[label]

        if label not in classNames:
            continue

        x1 = obj.bndbox.xmin/w
        y1 = obj.bndbox.ymin/h
        x2 = obj.bndbox.xmax/w
        y2 = obj.bndbox.ymax/h
        

        boxes.append([x1, y1, x2, y2])
        labels.append(classNames.index(label))

    if len(boxes) == 0:
        boxes = np.zeros((0, 4))

    return imgpath, boxes, labels, (w, h)



# =================================================
def cartoonize(image, *args, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.medianBlur(image, 1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 200, 200)
    cartoon = cv2.bitwise_and(color, color, mask = edges)
    return cartoon

# =================================================
def mixup(image, backgrounds=r'C:\Users\z004hkut\Downloads', alpha=0.7, *args, **kwargs):

    backgrounds = glob.glob(backgrounds+"**/**/**/*.jpg") + glob.glob(backgrounds+"**/**/*.jpg")

    bcgkpathImg = cv2.imread(random.choice(backgrounds))
    bcgkpathImg = cv2.resize(bcgkpathImg, (256, 256))
    bcgkpathImg = cv2.cvtColor(bcgkpathImg, cv2.COLOR_BGR2RGB)

    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, bcgkpathImg, beta, 0.0)
    return dst

# =================================================
def nope(x, *args, **kwargs):
    return x