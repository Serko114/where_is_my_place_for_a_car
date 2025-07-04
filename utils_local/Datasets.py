from utils_local.Datasets import *  # Dataset  # as BaseDataset
from utils_local.utils import get_validation_augmentation as augmentation
import numpy as np
import cv2


class Dataset:
    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
        preprocessing=None
    ):
        self.images_paths = images_dir  # glob(f"{images_dir}/*")
        # frame: (1080, 1920, 3) uni8
        self.masks_paths = masks_dir  # glob(f"{masks_dir}/*")
        self.LABEL_COLORS_FILE = 'data/1_sum_data_segmentation/label_colors.txt'
        self.cls_colors = self._get_classes_colors(self.LABEL_COLORS_FILE)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def _get_classes_colors(self, label_colors_dir):
        cls_colors = {}
        with open(label_colors_dir) as file:
            while line := file.readline():
                R, G, B, label = line.rstrip().split()
                cls_colors[label] = np.array([B, G, R], dtype=np.uint8)
        CLASSES = [
            "background",
            "ground",
            "no_car",
        ]
        keyorder = CLASSES
        cls_colors_ordered = {}
        for k in keyorder:
            if k in cls_colors:
                cls_colors_ordered[k] = cls_colors[k]
            elif k == "background":
                cls_colors_ordered[k] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                raise ValueError(
                    f"unexpected label {k}, cls colors: {cls_colors}")

        return cls_colors_ordered

    def __getitem__(self, i):
        # image = cv2.imread(self.images_paths[i])
        image = self.images_paths
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_paths)  # [i])
        masks = [cv2.inRange(mask, color, color)
                 for color in self.cls_colors.values()]
        masks = [(m > 0).astype("float32") for m in masks]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_paths)
