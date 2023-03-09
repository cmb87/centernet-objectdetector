import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import readJsonAnnotation, readPASCALVOCAnnotation



# ==================================
def createTrainingFile(path, classNames, imageSubPath="./", fileType="json", **kwargs):

    # Select File Parser
    fileParser = {
        "json": readJsonAnnotation,
        "pascalvoc": readPASCALVOCAnnotation
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


    classNames =  ["NC", "WF", "MR", "IN"]
    path = r"C:\Users\z004hkut\projects\01_robotics\01_ml\project-ctchr\objectDetector\data\stickytraps"
    imageSubPath = "./"

    df = createTrainingFile(path, classNames, imageSubPath, fileType="json")
    df.to_csv("test.csv", index=False)