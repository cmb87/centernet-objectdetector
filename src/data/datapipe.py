import tensorflow as tf
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,Crop ,
    Rotate, KeypointParams, Cutout, Superpixels, Spatter, Sharpen, OpticalDistortion, Affine, Perspective, FancyPCA, ToSepia,
    OneOf, RGBShift, ShiftScaleRotate, CenterCrop, VerticalFlip, RandomCrop, Lambda, BboxParams
)

from labels import encodeTF, drawTF


DEFAULTTRANSFORM = Compose([
    # Rotate(limit=50, p=0.5),
    # Lambda(image = cartoonize, keypoint=nope, bbox=nope, always_apply=False, p=1.0),
    ToSepia(always_apply=False, p=0.1),
    RandomContrast(limit=0.2, p=0.5),
    Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
    JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    OneOf([
        HueSaturationValue(p=0.5), 
        RGBShift(p=0.7),
        RandomBrightness(limit=0.1),  
    ], p=1), 

    # Superpixels(p_replace=0.5, n_segments=128, max_size=128, interpolation=1, always_apply=False, p=0.9),

    ShiftScaleRotate(scale_limit=[-0.3,0.3], shift_limit=[0.0,0.3], border_mode = cv2.BORDER_REPLICATE, p=0.5),
    HorizontalFlip(),
    VerticalFlip(),
    #Lambda(image = mixup, keypoint=nope, bbox=nope, always_apply=False, p=1.0),
    #Cutout(num_holes=18, max_h_size=18, max_w_size=18, fill_value=0, always_apply=False, p=0.3),
    #Spatter(mean=0.65, std=0.3, gauss_sigma=2, cutout_threshold=0.68, intensity=0.6, mode='rain', always_apply=False, p=0.3)
],
bbox_params=BboxParams(format='albumentations') #  , remove_invisible=True, angle_in_degrees=True
)



class Datapipe:

    # ============================
    def __init__(self):
        self.nx,self.ny = None, None
        self.iw,self.ih = None, None
        self.transform = None

    # ============================
    def _processLoadImage(self, imgPath, bboxes, labels):

        ih, iw = self.ih, self.iw
        ic = self.ic

        img = tf.io.read_file(imgPath)
        img = tf.image.decode_jpeg(img, channels=ic)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (ih, iw))

        return img, bboxes, labels


    # ============================
    def _processAugment(self, img, bboxes, labels):

        def aug_fn(img, bboxes, labels, iw, ih):

            boxes_ext = [b.tolist() + [l] for b,l in zip(bboxes,labels)]
            data = {"image": np.uint8(255*img), "bboxes":boxes_ext}

            aug_data = self.transform(**data)
            aug_img = aug_data["image"]
            aug_img = tf.cast(aug_img/255.0, tf.float32)
            aug_img = tf.image.resize(aug_img, size=[ih , iw ])

            bboxes = np.asarray([b[:4] for b in aug_data["bboxes"]]).astype(np.float32)
            labels = np.asarray([b[-1] for b in aug_data["bboxes"]]).astype(np.int32)

            return aug_img, bboxes, labels


        aug_img, bboxes, labels = tf.numpy_function(
            func=aug_fn,
            inp=[img, bboxes, labels, self.iw, self.ih],
            Tout=[tf.float32,tf.float32,tf.int32]
        )

        return aug_img, bboxes, labels


    # ============================
    @staticmethod
    def _processConvert(imgPath, bboxes, labels):

        bboxes, labels = tf.py_function(
            lambda x,y: (eval(x.numpy().decode("utf-8")), eval(y.numpy().decode("utf-8"))),
            [bboxes, labels],
            [tf.float32, tf.int32]
        )

        # Clip Bounding Boxes
        bboxes = tf.clip_by_value(bboxes,0.0,1.0)

        return imgPath, bboxes, labels

    # ============================
    def _gaussianLabel(self, img, bboxes, labels):
        
        bboxes = tf.clip_by_value(bboxes,0.0,1.0)
        labels = tf.one_hot(labels, depth=self.nc)
        y = encodeTF(bboxes, labels, self.nx, self.ny)
        

        return img, y


    # ============================
    def __call__(self, csvFiles, nx, ny, nc, iw, ih, ic, batchSize=3, sigma=0.02, shuffle_buffer_size=8000, nrepeat=1, augment=True, shuffle=True, transform=None):

        self.nx,self.ny = nx, ny
        self.iw,self.ih = iw, ih
        self.ic = ic
        self.nc = nc
        self.transform = DEFAULTTRANSFORM if transform is None else transform


        ds = tf.data.experimental.CsvDataset(csvFiles, [tf.string, tf.string, tf.string], select_cols=[1,2,3], header=True)
        ds = ds.shuffle(shuffle_buffer_size).repeat(nrepeat)
        ds = ds.map(Datapipe._processConvert)

        ds = ds.map(self._processLoadImage)

        if augment:
            ds = ds.map(self._processAugment)

        ds = ds.map(self._gaussianLabel)

        ds = ds.batch(batchSize)

        return ds





# ============================
if __name__ == "__main__":

    pipe = Datapipe()

    ih,iw,ic = 128*3, 128*4, 3
    ny,nx,nc = ih//4,iw//4,4


    ds = pipe(["test.csv"], nx,ny,nc,iw,ih,ic)


    for (x,y) in ds.take(13):
        
        imgsAug = drawTF(x, y, thres=0.1, normalizedImage=True)

        for b in range(x.shape[0]):
            plt.imshow(imgsAug[b,...])
            plt.show()