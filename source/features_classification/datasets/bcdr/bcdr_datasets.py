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


class BCDR_Pathology_Dataset(Dataset):
    classes = np.array(['BACKGROUND',
                        'BENIGN_MASS', 'MALIGNANT_MASS',
                        'BENIGN_CALC', 'MALIGNANT_CALC',
                        'BENIGN_MICROCALC', 'MALIGNANT_MICROCALC',
                        'BENIGN_MASS-CALC', 'MALIGNANT_MASS-CALC',
                        'BENIGN_MASS-MICROCALC', 'MALIGNANT_MASS-MICROCALC',
                        'BENIGN_CALC-MICROCALC', 'MALIGNANT_CALC-MICROCALC'
                        ])

        
    def __init__(self, mass_root_dir, calc_root_dir, microcalc_root_dir,
                 masscalc_root_dir, massmicrocalc_root_dir, calcmicrocalc_root_dir,
                 bg_root_dir, transform=None):
        self.transform = transform

        self.mass_root_dir = mass_root_dir
        self.calc_root_dir = calc_root_dir
        self.microcalc_root_dir = microcalc_root_dir

        self.masscalc_root_dir = masscalc_root_dir
        self.massmicrocalc_root_dir = massmicrocalc_root_dir
        self.calcmicrocalc_root_dir = calcmicrocalc_root_dir

        self.bg_root_dir = bg_root_dir

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(BCDR_Pathology_Dataset.classes):
            if class_name == 'BACKGROUND':
                bg_images = glob.glob(os.path.join(bg_root_dir, '*.png'))
                self.images_list += bg_images
                self.labels += [idx] * len(bg_images)

                if len(bg_images) == 0:
                    print(bg_root_dir)
                    raise ValueError
            else:
                pathology, lesion_type = class_name.split('_')

                if lesion_type == 'MASS':
                    mass_images = glob.glob(os.path.join(
                        mass_root_dir, pathology, '*.png'
                    ))
                    self.images_list += mass_images
                    self.labels += [idx] * len(mass_images)

                    # if len(mass_images) == 0:
                    #     raise ValueError
                elif lesion_type == 'CALC':
                    calc_images = glob.glob(os.path.join(
                        calc_root_dir, pathology, '*.png'
                    ))
                    self.images_list += calc_images
                    self.labels += [idx] * len(calc_images)

                    # if len(calc_images) == 0:
                    #     raise ValueError
                elif lesion_type == 'MICROCALC':
                    microcalc_images = glob.glob(os.path.join(
                        microcalc_root_dir, pathology, '*.png'
                    ))
                    self.images_list += microcalc_images
                    self.labels += [idx] * len(microcalc_images)

                    # if len(microcalc_images) == 0:
                    #     raise ValueError
                elif lesion_type == 'MASS-CALC':
                    masscalc_images = glob.glob(os.path.join(
                        masscalc_root_dir, pathology, '*.png'
                    ))
                    self.images_list += masscalc_images
                    self.labels += [idx] * len(masscalc_images)

                    if len(masscalc_images) == 0:
                        raise ValueError
                elif lesion_type == 'MASS-MICROCALC':
                    massmicrocalc_images = glob.glob(os.path.join(
                        massmicrocalc_root_dir, pathology, '*.png'
                    ))
                    self.images_list += massmicrocalc_images
                    self.labels += [idx] * len(massmicrocalc_images)

                    # if len(massmicrocalc_images) == 0:
                    #     raise ValueError
                elif lesion_type == 'CALC-MICROCALC':
                    calcmicrocalc_images = glob.glob(os.path.join(
                        calcmicrocalc_root_dir, pathology, '*.png'
                    ))
                    self.images_list += calcmicrocalc_images
                    self.labels += [idx] * len(calcmicrocalc_images)

                    # if len(calcmicrocalc_images) == 0:
                    #     raise ValueError

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


class BCDR_Pathology_RandomCrops_Dataset(Dataset):
    classes = np.array(['BACKGROUND'])

        
    def __init__(self, bg_root_dir, transform=None):
        self.transform = transform

        self.bg_root_dir = bg_root_dir

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(BCDR_Pathology_RandomCrops_Dataset.classes):
            if class_name == 'BACKGROUND':
                bg_images = glob.glob(os.path.join(bg_root_dir, '*.png'))
                self.images_list += bg_images
                self.labels += [idx] * len(bg_images)

                if len(bg_images) == 0:
                    print(bg_root_dir)
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


class All_BCDR_Pathology_Dataset(Dataset):
    classes = np.array(['BACKGROUND',
                        'BENIGN_MASS', 'MALIGNANT_MASS',
                        'BENIGN_CALC', 'MALIGNANT_CALC',
                        'BENIGN_MICROCALC', 'MALIGNANT_MICROCALC',
                        'BENIGN_MASS-CALC', 'MALIGNANT_MASS-CALC',
                        'BENIGN_MASS-MICROCALC', 'MALIGNANT_MASS-MICROCALC',
                        'BENIGN_CALC-MICROCALC', 'MALIGNANT_CALC-MICROCALC'
                        ])

        
    def __init__(self, film_root, digital_root, data_type='pathology',
                 transform=None, use_dn01=False):
        self.transform = transform

        F01_cls_root = os.path.join(film_root, 'BCDR-F01_dataset', 'cls')
        F01_dataset = BCDR_Pathology_Dataset(
            os.path.join(F01_cls_root, 'mass', data_type),
            os.path.join(F01_cls_root, 'calc', data_type),
            os.path.join(F01_cls_root, 'microcalc', data_type),
            os.path.join(F01_cls_root, 'mass_calc', data_type),
            os.path.join(F01_cls_root, 'mass_microcalc', data_type),
            os.path.join(F01_cls_root, 'calc_microcalc', data_type),
            os.path.join(F01_cls_root, 'background', 'background_tfds'),
            transform=transform
        )

        F02_cls_root = os.path.join(film_root, 'BCDR-F02_dataset', 'cls')
        F02_dataset = BCDR_Pathology_Dataset(
            os.path.join(F02_cls_root, 'mass', data_type),
            os.path.join(F02_cls_root, 'calc', data_type),
            os.path.join(F02_cls_root, 'microcalc', data_type),
            os.path.join(F02_cls_root, 'mass_calc', data_type),
            os.path.join(F02_cls_root, 'mass_microcalc', data_type),
            os.path.join(F02_cls_root, 'calc_microcalc', data_type),
            os.path.join(F02_cls_root, 'background', 'background_tfds'),
            transform=transform
        )

        F03_cls_root = os.path.join(film_root, 'BCDR-F03_dataset', 'cls')
        F03_dataset = BCDR_Pathology_Dataset(
            os.path.join(F03_cls_root, 'mass', data_type),
            os.path.join(F03_cls_root, 'calc', data_type),
            os.path.join(F03_cls_root, 'microcalc', data_type),
            os.path.join(F03_cls_root, 'mass_calc', data_type),
            os.path.join(F03_cls_root, 'mass_microcalc', data_type),
            os.path.join(F03_cls_root, 'calc_microcalc', data_type),
            os.path.join(F03_cls_root, 'background', 'background_tfds'),
            transform=transform
        )

        D01_cls_root = os.path.join(digital_root, 'BCDR-D01_dataset', 'cls')
        D01_dataset = BCDR_Pathology_Dataset(
            os.path.join(D01_cls_root, 'mass', data_type),
            os.path.join(D01_cls_root, 'calc', data_type),
            os.path.join(D01_cls_root, 'microcalc', data_type),
            os.path.join(D01_cls_root, 'mass_calc', data_type),
            os.path.join(D01_cls_root, 'mass_microcalc', data_type),
            os.path.join(D01_cls_root, 'calc_microcalc', data_type),
            os.path.join(D01_cls_root, 'background', 'background_tfds'),
            transform=transform
        )

        D02_cls_root = os.path.join(digital_root, 'BCDR-D02_dataset', 'cls')
        D02_dataset = BCDR_Pathology_Dataset(
            os.path.join(D02_cls_root, 'mass', data_type),
            os.path.join(D02_cls_root, 'calc', data_type),
            os.path.join(D02_cls_root, 'microcalc', data_type),
            os.path.join(D02_cls_root, 'mass_calc', data_type),
            os.path.join(D02_cls_root, 'mass_microcalc', data_type),
            os.path.join(D02_cls_root, 'calc_microcalc', data_type),
            os.path.join(D02_cls_root, 'background', 'background_tfds'),
            transform=transform
        )

        self.images_list = \
            F01_dataset.get_images_list() + \
            F02_dataset.get_images_list() + \
            F03_dataset.get_images_list() + \
            D01_dataset.get_images_list() + \
            D02_dataset.get_images_list()

        if len(F01_dataset.get_images_list()) == 0:
            raise ValueError
        if len(F02_dataset.get_images_list()) == 0:
            raise ValueError
        if len(F03_dataset.get_images_list()) == 0:
            raise ValueError
        if len(D01_dataset.get_images_list()) == 0:
            raise ValueError
        if len(D02_dataset.get_images_list()) == 0:
            raise ValueError

        self.labels = \
            F01_dataset.get_labels() + \
            F02_dataset.get_labels() + \
            F03_dataset.get_labels() + \
            D01_dataset.get_labels() + \
            D02_dataset.get_labels()

        if use_dn01:
            DN01_cls_root = os.path.join(digital_root, 'BCDR-DN01_dataset', 'cls')
            DN01_dataset = BCDR_Pathology_RandomCrops_Dataset(
                os.path.join(DN01_cls_root, 'background', 'background_tfds')
            )
            if len(DN01_dataset.get_images_list()) == 0:
                raise ValueError

            self.images_list = self.images_list + DN01_dataset.get_images_list()
            self.labels = self.labels + DN01_dataset.get_labels()

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
