import tensorflow as tf
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, HorizontalFlip,Crop ,
    Rotate, KeypointParams, CoarseDropout, Superpixels, Spatter, Sharpen, OpticalDistortion, Affine, Perspective, FancyPCA, ToSepia,
    OneOf, RGBShift, ShiftScaleRotate, CenterCrop, VerticalFlip, RandomCrop, Lambda, BboxParams,ToGray, GaussNoise, ISONoise 
)

from .labels import encodeTF, drawTF


DEFAULTTRANSFORM = Compose([
    # Rotate(limit=50, p=0.5),
    # Lambda(image = cartoonize, keypoint=nope, bbox=nope, always_apply=False, p=1.0),
    ToSepia(p=0.1),

    ToGray(p=0.2),
    Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
   # JpegCompression(quality_lower=85, quality_upper=100, p=0.3),
    OneOf([
        HueSaturationValue(p=0.3), 
        RGBShift(p=0.3),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), p=0.8),  
    ], p=1), 

    # Superpixels(p_replace=0.5, n_segments=128, max_size=128, interpolation=1, always_apply=False, p=0.9),

    ShiftScaleRotate(scale_limit=[-0.15,0.15], shift_limit=[0.0,0.15], border_mode = cv2.BORDER_REPLICATE, p=0.5),
    HorizontalFlip(),
    VerticalFlip(),
    OneOf([
      ISONoise(p=0.1),
      GaussNoise(p=0.1),
    ], p=1), 

    #Lambda(image = mixup, keypoint=nope, bbox=nope, always_apply=False, p=1.0),
    CoarseDropout(num_holes_range=(3, 6),hole_height_range=(10, 100),hole_width_range=(10, 100),fill="random_uniform", p=0.3),
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
        #return img, bboxes, labels, imgPath


    # ============================
    # ============================
    def _processAugment(self, img, bboxes, labels, *args):

        def aug_fn(img, bboxes, labels, iw, ih):
            import random
            
            img_uint8 = np.uint8(255 * img)
            boxes_ext = [b.tolist() for b in bboxes]
            labels_ext = labels.tolist()
            
            data = {
                "image": img_uint8,
                "bboxes": boxes_ext,
                "labels": labels_ext
            }
            
            # --- Dynamic Mosaic Preparation ---
            transform_cfg = getattr(self, "transform_cfg", None)
            if transform_cfg and transform_cfg.get("mosaic_p", 0.0) > 0 and random.random() < transform_cfg.get("mosaic_p", 0.0):
                mosaic_metadata = []
                all_records = getattr(self, "all_records", [])
                if len(all_records) >= 3:
                    sampled = random.sample(all_records, 3)
                    for s_path, s_bboxes_str, s_labels_str in sampled:
                        try:
                            s_img = cv2.imread(s_path)
                            if s_img is not None:
                                s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
                                s_img = cv2.resize(s_img, (iw, ih))
                                s_bboxes = eval(s_bboxes_str)
                                s_labels = eval(s_labels_str)
                                mosaic_metadata.append({
                                    'image': s_img,
                                    'bboxes': s_bboxes,
                                    'labels': s_labels
                                })
                        except Exception as e:
                            pass
                if mosaic_metadata:
                    data["mosaic_metadata"] = mosaic_metadata
            
            # Decide if we are using the new label_fields parameter scheme or the legacy append-style format
            is_using_label_fields = hasattr(self, "transform") and self.transform is not None and "labels" in self.transform.processors
            if not is_using_label_fields:
                data_old = {
                    "image": img_uint8,
                    "bboxes": [b.tolist() + [l] for b, l in zip(bboxes, labels)]
                }
                aug_data = self.transform(**data_old)
            else:
                aug_data = self.transform(**data)
                
            aug_img = aug_data["image"]
            
            if "labels" in aug_data:
                aug_boxes = list(aug_data["bboxes"])
                aug_labels = list(aug_data["labels"])
            else:
                aug_boxes = [b[:4] for b in aug_data["bboxes"]]
                aug_labels = [b[-1] for b in aug_data["bboxes"]]
                
            # --- Dynamic MixUp Augmentation ---
            if transform_cfg and transform_cfg.get("mixup_p", 0.0) > 0 and random.random() < transform_cfg.get("mixup_p", 0.0):
                all_records = getattr(self, "all_records", [])
                if len(all_records) >= 1:
                    s_path, s_bboxes_str, s_labels_str = random.choice(all_records)
                    try:
                        s_img = cv2.imread(s_path)
                        if s_img is not None:
                            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
                            s_img = cv2.resize(s_img, (iw, ih))
                            s_bboxes = eval(s_bboxes_str)
                            s_labels = eval(s_labels_str)
                            
                            # Blend images
                            lam = np.random.beta(1.0, 1.0)
                            aug_img = (lam * aug_img + (1.0 - lam) * s_img).astype(np.uint8)
                            
                            # Merge labels and bboxes
                            aug_boxes = list(aug_boxes) + s_bboxes
                            aug_labels = list(aug_labels) + s_labels
                    except Exception as e:
                        pass
                        
            # Resize and scale back to float32
            aug_img = tf.cast(aug_img / 255.0, tf.float32)
            aug_img = tf.image.resize(aug_img, size=[ih, iw])
            
            # Format outputs
            bboxes_out = np.asarray(aug_boxes).astype(np.float32)
            labels_out = np.asarray(aug_labels).astype(np.int32)
            
            if len(bboxes_out.shape) > 1 and bboxes_out.shape[0] > 0:
                xm = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])
                ym = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])
                idx = (xm >= 0.0) & (xm < 1.0) & (ym >= 0.0) & (ym < 1.0)
                return aug_img, bboxes_out[idx, :], labels_out[idx]
                
            return aug_img, bboxes_out, labels_out

        aug_img, bboxes, labels = tf.numpy_function(
            func=aug_fn,
            inp=[img, bboxes, labels, self.iw, self.ih],
            Tout=[tf.float32, tf.float32, tf.int32]
        )

        return aug_img, bboxes, labels, *args


    # ============================
    @staticmethod
    def _processConvert(imgPath, bboxes, labels, *args):

        bboxes, labels = tf.py_function(
            lambda x,y: (eval(x.numpy().decode("utf-8")), eval(y.numpy().decode("utf-8"))),
            [bboxes, labels],
            [tf.float32, tf.int32]
        )

        # Clip Bounding Boxes
        bboxes = tf.clip_by_value(bboxes,0.0,0.999)

        return imgPath, bboxes, labels, *args

    # ============================
    def _gaussianLabel(self, img, bboxes, labels):
        
        bboxes = tf.clip_by_value(bboxes,0.0,0.999)
        labels = tf.one_hot(labels, depth=self.nc)
        y = encodeTF(bboxes, labels, self.nx, self.ny)
        
        return img, y

    # ============================
    def _filter(self, img, bboxes, labels, *args):
        return tf.math.not_equal(tf.shape(bboxes)[0], 0)
    
    # ============================
    def _makeDataSet(self, csvFile, buffer=3000):
        ds = tf.data.experimental.CsvDataset([csvFile], [tf.string, tf.string, tf.string], select_cols=[1,2,3], header=True)
        ds = ds.shuffle(buffer).repeat(1)
        return ds
    
    # ============================
    # ============================
    def __call__(self, csvFiles, nx, ny, nc, iw, ih, ic, batchSize=3, sigma=0.02, shuffle_buffer_size=8000, nrepeat=1, augment=True, shuffle=True, transform=None, transform_cfg=None):

        self.nx,self.ny = nx, ny
        self.iw,self.ih = iw, ih
        self.ic = ic
        self.nc = nc
        self.transform_cfg = transform_cfg
        
        # Parse CSV files for dynamic Mosaic and MixUp loading
        self.all_records = []
        import csv
        import os
        for csvFile in csvFiles:
            if os.path.exists(csvFile):
                try:
                    with open(csvFile, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row = {k.strip() if k is not None else k: v for k, v in row.items()}
                            img_path = row.get("imagePath") or row.get("File") or row.get("filename")
                            bboxes_str = row.get("bboxes")
                            labels_str = row.get("labels")
                            if img_path and bboxes_str and labels_str:
                                self.all_records.append((img_path, bboxes_str, labels_str))
                except Exception as e:
                    print(f"Warning: Failed to load records from {csvFile} for Mosaic/MixUp: {e}")
        
        if transform is not None:
            self.transform = transform
        elif transform_cfg is not None:
            import albumentations as A
            import cv2
            
            is_thermal = transform_cfg.get("thermal_mode", False)
            transforms = []
            
            # 0. Mosaic (multi-image combination) - YOLO style
            if transform_cfg.get("mosaic_p", 0.0) > 0:
                transforms.append(A.Mosaic(
                    grid_yx=(2, 2),
                    target_size=(ih, iw),
                    cell_shape=(ih // 2, iw // 2),
                    center_range=(0.4, 0.6),
                    fit_mode="cover",
                    p=transform_cfg.mosaic_p
                ))
            
            # 1. Multi-scale detail zooming / Random Crop (boosts small-object resolution)
            if transform_cfg.get("random_crop_p", 0) > 0:
                transforms.append(A.RandomResizedCrop(
                    size=(ih, iw), 
                    scale=(0.5, 1.0), 
                    p=transform_cfg.random_crop_p
                ))
            
            # 2. Base Contrast & Sharpness (safe and useful for both thermal and RGB)
            if transform_cfg.get("sharpen_p", 0) > 0:
                transforms.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=transform_cfg.sharpen_p))
            if transform_cfg.get("motion_blur_p", 0) > 0:
                transforms.append(A.MotionBlur(p=transform_cfg.motion_blur_p))
                
            # 3. Color shifting - ONLY active when NOT in thermal mode
            if not is_thermal:
                if transform_cfg.get("to_sepia_p", 0) > 0:
                    transforms.append(A.ToSepia(p=transform_cfg.to_sepia_p))
                if transform_cfg.get("to_gray_p", 0) > 0:
                    transforms.append(A.ToGray(p=transform_cfg.to_gray_p))
                    
                one_of_color = []
                if transform_cfg.get("hsv_p", 0) > 0:
                    one_of_color.append(A.HueSaturationValue(p=transform_cfg.hsv_p))
                if transform_cfg.get("rgb_shift_p", 0) > 0:
                    one_of_color.append(A.RGBShift(p=transform_cfg.rgb_shift_p))
                if one_of_color:
                    transforms.append(A.OneOf(one_of_color, p=1.0))
            
            # 4. Thermal-safe brightness range modulator
            if transform_cfg.get("brightness_contrast_p", 0) > 0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.3), 
                    contrast_limit=(-0.2, 0.3), 
                    p=transform_cfg.brightness_contrast_p
                ))
                
            # 5. Spatial geometry modifications
            if transform_cfg.get("shift_scale_rotate_p", 0) > 0:
                transforms.append(A.ShiftScaleRotate(
                    scale_limit=[-0.15, 0.15], 
                    shift_limit=[0.0, 0.15], 
                    border_mode=cv2.BORDER_REPLICATE, 
                    p=transform_cfg.shift_scale_rotate_p
                ))
                
            if transform_cfg.get("horizontal_flip_p", 0) > 0:
                transforms.append(A.HorizontalFlip(p=transform_cfg.horizontal_flip_p))
            if transform_cfg.get("vertical_flip_p", 0) > 0:
                transforms.append(A.VerticalFlip(p=transform_cfg.vertical_flip_p))
                
            # 6. Low-level sensor static modelings (Noise)
            one_of_noise = []
            if transform_cfg.get("iso_noise_p", 0) > 0:
                one_of_noise.append(A.ISONoise(p=transform_cfg.iso_noise_p))
            if transform_cfg.get("gauss_noise_p", 0) > 0:
                one_of_noise.append(A.GaussNoise(p=transform_cfg.gauss_noise_p))
            if one_of_noise:
                transforms.append(A.OneOf(one_of_noise, p=1.0))
                
            # 7. Coarse information dropping (Regularization)
            if transform_cfg.get("coarse_dropout_p", 0) > 0:
                transforms.append(A.CoarseDropout(
                    num_holes_range=(3, 6), 
                    hole_height_range=(10, 100), 
                    hole_width_range=(10, 100), 
                    fill="random_uniform", 
                    p=transform_cfg.coarse_dropout_p
                ))
                
            # 8. Filter out cut-off boxes or zero-pixel labels dynamically
            min_area = transform_cfg.get("min_area", 0)
            min_visibility = transform_cfg.get("min_visibility", 0.0)
            
            self.transform = A.Compose(
                transforms, 
                bbox_params=A.BboxParams(
                    format='albumentations',
                    min_area=min_area,
                    min_visibility=min_visibility,
                    label_fields=['labels']
                )
            )
        else:
            self.transform = DEFAULTTRANSFORM

        # https://stackoverflow.com/questions/54843448/how-to-zip-tensorflow-dataset-and-train-in-keras-correctly
        # https://stackoverflow.com/questions/64725275/how-to-configure-dataset-pipelines-with-tensorflow-make-csv-dataset-for-keras-mo

        #ds = tf.data.experimental.CsvDataset(csvFiles, [tf.string, tf.string, tf.string], select_cols=[1,2,3], header=True)
        #ds = ds.shuffle(shuffle_buffer_size).repeat(nrepeat)

        # # Resample (Oversample from different class represented too few)
        weights = len(csvFiles)*[1.0/len(csvFiles)]

        if len(csvFiles) > 1:
            dss = [self._makeDataSet(csvFile, shuffle_buffer_size) for csvFile in csvFiles]
            ds = tf.data.Dataset.sample_from_datasets(dss, weights=weights)
        else:
            ds = self._makeDataSet(csvFiles[0], shuffle_buffer_size)


        ds = ds.map(Datapipe._processConvert)

        ds = ds.map(self._processLoadImage)

        if augment:
            ds = ds.map(self._processAugment)

        ds = ds.filter(self._filter)

        # See https://stackoverflow.com/questions/62585490/as-list-is-not-defined-on-an-unknown-tensorshape-on-y-t-rank-leny-t-shape-a
        def _fixup_shape(x, y):
            x.set_shape([None, None, None, 3])
            y.set_shape([None, None, None, self.nc + 4])
            return x, y

        ds = ds.map(self._gaussianLabel)
        ds = ds.repeat()
        ds = ds.batch(batchSize)
        ds = ds.map(_fixup_shape)
        ds = ds.prefetch(tf.data.AUTOTUNE)
  
        return ds



# ============================
if __name__ == "__main__":

    pipe = Datapipe()

    ih,iw,ic = 128*4, 128*4, 3
    ny,nx,nc = ih//4,iw//4,4


    ds = pipe(["../thermalDet_train.csv"], nx,ny,nc,iw,ih,ic)

    for (x,y) in ds.take(13):
        
        imgsAug = drawTF(x, y, thres=0.1, normalizedImage=True)

        print(x.shape, y.shape)

        if True:
  
            for b in range(x.shape[0]):
               # plt.title(p[b])
                plt.imshow(imgsAug[b,...])
                plt.savefig(f"augmented_{b}.png")
                plt.show()
            