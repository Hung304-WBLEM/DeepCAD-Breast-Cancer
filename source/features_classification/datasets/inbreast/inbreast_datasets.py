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
# from dataprocessing.process_cbis_ddsm import get_info_lesion
from sklearn.preprocessing import label_binarize
from natsort import natsorted

from features_classification.train.train_utils import compute_classes_weights


class INBreast_Pathology_Dataset(Dataset):
    classes = np.array(['BACKGROUND',
                        'BENIGN_MASS', 'MALIGNANT_MASS',
                        'BENIGN_CALC', 'MALIGNANT_CALC',
                        'BENIGN_SPICULATED', 'MALIGNANT_SPICULATED',
                        'MALIGNANT_ASYMETRY',
                        'BENIGN_DISTORTION', 'MALIGNANT_DISTORTION',
                        'BENIGN_CLUSTER', 'MALIGNANT_CLUSTER'
                        ])

    def __init__(self, mass_root_dir, calc_root_dir,
                 spiculated_root_dir, asymetry_root_dir,
                 distortion_root_dir, cluster_root_dir,
                 bg_root_dir, transform=None):
        self.transform = transform
        self.mass_root_dir = mass_root_dir
        self.calc_root_dir = calc_root_dir
        self.spiculated_root_dir = spiculated_root_dir
        self.asymetry_root_dir = asymetry_root_dir
        self.distortion_root_dir = distortion_root_dir
        self.cluster_root_dir = cluster_root_dir
        self.bg_root_dir = bg_root_dir

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(INBreast_Pathology_Dataset.classes):
            if class_name == 'BACKGROUND':
                bg_images = glob.glob(os.path.join(bg_root_dir, '*.png'))
                self.images_list += bg_images
                self.labels += [idx] * len(bg_images)

                if len(bg_images) == 0:
                    raise ValueError
            else:
                
                pathology, lesion_type = class_name.split('_')

                if lesion_type == 'MASS':
                    mass_images = glob.glob(os.path.join(
                        mass_root_dir, pathology, '*.png'))
                    self.images_list += mass_images
                    self.labels += [idx] * len(mass_images)

                    if len(mass_images) == 0:
                        raise ValueError
                elif lesion_type == 'CALC':
                    calc_images = glob.glob(os.path.join(
                        calc_root_dir, pathology, '*.png'))
                    self.images_list += calc_images
                    self.labels += [idx] * len(calc_images)

                    if len(calc_images) == 0:
                        raise ValueError
                elif lesion_type == 'SPICULATED':
                    spiculated_images = glob.glob(os.path.join(
                        spiculated_root_dir, pathology, '*.png'))
                    self.images_list += spiculated_images
                    self.labels += [idx] * len(spiculated_images)

                    if len(spiculated_images) == 0:
                        raise ValueError
                elif lesion_type == 'ASYMETRY':
                    asymetry_images = glob.glob(os.path.join(
                        asymetry_root_dir, pathology, '*.png'))
                    self.images_list += asymetry_images
                    self.labels += [idx] * len(asymetry_images)

                    if len(asymetry_images) == 0:
                        raise ValueError
                elif lesion_type == 'DISTORTION':
                    distortion_images = glob.glob(os.path.join(
                        distortion_root_dir, pathology, '*.png'))
                    self.images_list += distortion_images
                    self.labels += [idx] * len(distortion_images)

                    if len(distortion_images) == 0:
                        raise ValueError
                elif lesion_type == 'CLUSTER':
                    cluster_images = glob.glob(os.path.join(
                        cluster_root_dir, pathology, '*.png'))
                    self.images_list += cluster_images
                    self.labels += [idx] * len(cluster_images)

                    if len(cluster_images) == 0:
                        raise ValueError

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
