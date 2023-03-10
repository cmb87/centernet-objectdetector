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

    name = "sticktraps"
    path = "/SHARE4ALL/demoData/stickytraps"
    #name = "synthetic"
    #path = "/SHARE4ALL/demoData/synthetic"


    classNames =  ["NC", "WF", "MR", "IN"]

    imageSubPath = "./"

    df = createTrainingFile(path, classNames, imageSubPath, fileType="json", renameDict={"TR": "IN"})

    train, test = train_test_split(df, test_size=0.2)

    test.to_csv(f"{name}_test.csv",index=False)
    train.to_csv(f"{name}_train.csv",index=False)

