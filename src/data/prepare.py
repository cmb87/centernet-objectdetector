import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from utils import readJsonAnnotation, readPASCALVOCAnnotation



# ==================================
def createTrainingFile(path, classNames, imageSubPath="./", fileType="json", **kwargs):

    # Select File Parser
    fileParser = {
        "json": readJsonAnnotation,
        "xml": readPASCALVOCAnnotation
    }[fileType]


    df = pd.DataFrame(columns=["id", "imagePath", "bboxes", "labels", "size"])


    for ctr,f in enumerate(tqdm(glob.glob(os.path.join(path, f"*.{fileType}")))):

        datapath = os.path.join(path,imageSubPath)
        imgpathAbsolute, boxes, labels, size = fileParser(f, datapath, classNames, **kwargs)

        
        df = pd.concat((
            df,
            pd.DataFrame({'id':ctr,'imagePath': [imgpathAbsolute], 'bboxes': [boxes], "labels": [labels], "size": [size]})
        ))

        df.reset_index()

    return df




if __name__ == "__main__":

    # classNames =  ["NC", "WF", "MR", "IN"]
    # imageSubPath = "./"
    # fileType = "json"
    # renameDict={"TR": "IN"}
    # name = "outdoor1Train"
    # path = "/SHARE4ALL/testData/outdoor1Train"


    #classNames =  ["NC", "WF", "MR", "IN"]
    #imageSubPath = "./"
    #fileType = "json"
    #renameDict={"TR": "IN"}
    #name = "outdoor2Train"
    #path = "/SHARE4ALL/testData/outdoor2Train"

   # classNames =  ["NC", "WF", "MR", "IN"]
    #imageSubPath = "./"
   # fileType = "json"
   # renameDict={"TR": "IN"}
   # name = "farmer1Train"
   # path = "/SHARE4ALL/testData/farmer1Train"

    # classNames =  ["NC", "WF", "MR", "IN", "TR"]
    # imageSubPath = "./"
    # fileType = "json"
    # renameDict={} #{"TR": "IN"}
    # name = "stickytraps"
    # path = "/SHARE4ALL/testData/stickytraps"

    # classNames =  ["NC", "WF", "MR", "IN", "TR"]
    # imageSubPath = "./"
    # fileType = "json"
    # renameDict={} #{"TR": "IN"}
    # name = "farmer1Tiny"
    # path = "/SHARE4ALL/testData/farmer1Tiny"
 


    # classNames =  ["NC", "WF", "MR", "TR"]
    # imageSubPath = "./"
    # fileType = "json"
    # renameDict={} #{"TR": "IN"}
    # name = "stickytrapsWOIn"
    # path = "/SHARE4ALL/testData/stickytraps"

    classNames =  ["NC", "WF", "MR", "TR"]
    imageSubPath = "./"
    fileType = "json"
    renameDict={} #{"TR": "IN"}
    name = "farmer1WOIn"
    path = "/SHARE4ALL/testData/farmer1Tiny"
 

    if False:
        name = "VOC2007p12"
        #name = "VOC2007"
        renameDict={}
        classNames =  ["person","bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable","pottedplant", "sofa", "tvmonitor"]
        imageSubPath = "../"
        fileType = "xml"

        path =     "/SHARE4ALL/pascalVOC/VOC2007p12/Annotations"
        #path = "/SHARE4ALL/pascalVOC/VOC2007/test/VOCdevkit/VOC2007/Annotations"

        
    

    df = createTrainingFile(path, classNames, imageSubPath, fileType=fileType, renameDict=renameDict)

    train, test = train_test_split(df, test_size=0.1)

    test.to_csv(f"{name}_test.csv",index=False)
    train.to_csv(f"{name}_train.csv",index=False)

