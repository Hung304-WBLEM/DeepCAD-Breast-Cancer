import os
import glob
import pandas as pd
import torch
import numpy as np
import math
import random
import cv2
import albumentations

from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import label_binarize
from natsort import natsorted

from features_classification.train.train_utils import compute_classes_weights


class CSAWM_Dataset(Dataset):
    classes = np.array(['BENIGN',
                        'MALIGNANT'])

    def __init__(self, data_root_dir, transform=None):
        self.transform = transform
        self.data_root_dir = data_root_dir

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(CSAWM_Dataset.classes):
            if class_name == 'BENIGN':
                images = glob.glob(os.path.join(self.data_root_dir, 'BENIGN', '*.png'))
            elif class_name == 'MALIGNANT':
                images = glob.glob(os.path.join(self.data_root_dir, 'MALIGNANT', '*.png'))

            if len(images) == 0:
                raise ValueError

            self.images_list += images
            self.labels += [idx] * len(images)

    def get_images_list(self):
        return self.images_list

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)

        if image.mode == 'L':
            image = image.convert("RGB")

        label = self.labels[idx]

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)


        return {'image': image, 'label': label, 'img_path': img_path}
