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
from dataprocessing.process_cbis_ddsm import get_info_lesion
from sklearn.preprocessing import label_binarize


class Mass_Shape_Dataset(Dataset):
    classes = np.array(['ROUND', 'OVAL', 'IRREGULAR',
                        'LOBULATED', 'ARCHITECTURAL_DISTORTION',
                        'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE',
                        'FOCAL_ASYMMETRIC_DENSITY'])

    @staticmethod
    def convert_label_to_multilabel(class_name):
        multilabel = (Mass_Shape_Dataset.classes == class_name).astype(float)
        return multilabel

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, mass_shape in enumerate(Mass_Shape_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, mass_shape, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]
        binarized_multilabel = Mass_Shape_Dataset.convert_label_to_multilabel(Mass_Shape_Dataset.classes[label])

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)

        return {'image': image, 'label': label, 'binarized_multilabel': binarized_multilabel, 'img_path': img_path}


class Mass_Margins_Dataset(Dataset):
    classes = np.array(['ILL_DEFINED',
                        'CIRCUMSCRIBED',
                        'SPICULATED',
                        'MICROLOBULATED',
                        'OBSCURED'])

    @staticmethod
    def convert_label_to_multilabel(class_name):
        multilabel = (Mass_Margins_Dataset.classes == class_name).astype(float)
        return multilabel

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, mass_margins in enumerate(Mass_Margins_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, mass_margins, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]
        binarized_multilabel = Mass_Margins_Dataset.convert_label_to_multilabel(Mass_Margins_Dataset.classes[label])

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)

        return {'image': image, 'label': label, 'binarized_multilabel': binarized_multilabel, 'img_path': img_path}


class Calc_Type_Dataset(Dataset):
    classes = np.array(["AMORPHOUS", "PUNCTATE", "VASCULAR",
                        "LARGE_RODLIKE", "DYSTROPHIC", "SKIN",
                        "MILK_OF_CALCIUM", "EGGSHELL", "PLEOMORPHIC",
                        "COARSE", "FINE_LINEAR_BRANCHING",
                        "LUCENT_CENTER", "ROUND_AND_REGULAR",
                        "LUCENT_CENTERED"])

    @staticmethod
    def convert_label_to_multilabel(class_name):
        multilabel = (Calc_Type_Dataset.classes == class_name).astype(float)
        return multilabel

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, calc_type in enumerate(Calc_Type_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, calc_type, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]
        binarized_multilabel = Calc_Type_Dataset.convert_label_to_multilabel(Calc_Type_Dataset.classes[label])

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)

        return {'image': image, 'label': label, 'binarized_multilabel': binarized_multilabel, 'img_path': img_path}


class Calc_Dist_Dataset(Dataset):
    classes = np.array(["CLUSTERED",
                       "LINEAR",
                       "REGIONAL",
                       "DIFFUSELY_SCATTERED",
                       "SEGMENTAL"])

    @staticmethod
    def convert_label_to_multilabel(class_name):
        multilabel = (Calc_Dist_Dataset.classes == class_name).astype(float)
        return multilabel

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, calc_dist in enumerate(Calc_Dist_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, calc_dist, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]
        binarized_multilabel = Calc_Dist_Dataset.convert_label_to_multilabel(Calc_Dist_Dataset.classes[label])

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)

        return {'image': image, 'label': label, 'binarized_multilabel': binarized_multilabel, 'img_path': img_path}


class Breast_Density_Dataset(Dataset):
    classes = np.array(['1', '2', '3', '4'])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, breast_density in enumerate(Breast_Density_Dataset.classes):
            images = glob.glob(os.path.join(
                root_dir, str(breast_density), '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'img_path': img_path}


class Pathology_Dataset(Dataset):
    classes = np.array(['BENIGN', 'MALIGNANT'])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, pathology in enumerate(Pathology_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, pathology, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'img_path': img_path}


class Mass_Calc_Pathology_Dataset(Dataset):
    classes = np.array(['BENIGN', 'MALIGNANT'])

    def __init__(self, mass_root_dir, calc_root_dir, transform=None):
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, pathology in enumerate(Mass_Calc_Pathology_Dataset.classes):
            mass_images = glob.glob(os.path.join(
                mass_root_dir, pathology, '*.png'))
            self.images_list += mass_images
            self.labels += [idx] * len(mass_images)

            calc_images = glob.glob(os.path.join(
                calc_root_dir, pathology, '*.png'))
            self.images_list += calc_images
            self.labels += [idx] * len(calc_images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'img_path': img_path}


class Four_Classes_Mass_Calc_Pathology_Dataset(Dataset):
    classes = np.array(['BENIGN_MASS', 'MALIGNANT_MASS', 'BENIGN_CALC', 'MALIGNANT_CALC'])

    def __init__(self, mass_root_dir, calc_root_dir, transform=None):
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(Four_Classes_Mass_Calc_Pathology_Dataset.classes):
            pathology, lesion_type = class_name.split('_')

            if lesion_type == 'MASS':
                mass_images = glob.glob(os.path.join(
                    mass_root_dir, pathology, '*.png'))
                self.images_list += mass_images
                self.labels += [idx] * len(mass_images)
            elif lesion_type == 'CALC':
                calc_images = glob.glob(os.path.join(
                    calc_root_dir, pathology, '*.png'))
                self.images_list += calc_images
                self.labels += [idx] * len(calc_images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))

        image = Image.open(img_path)

        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # For Histogram Equalization
        # image = cv2.equalizeHist(image)

        # For CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)

        # duplicate channels
        # image = np.stack([image, image, image], axis=2)
        # image = Image.fromarray(image)

        label = self.labels[idx]

        if self.transform:
            if isinstance(self.transform, albumentations.core.composition.Compose):
                res = self.transform(image=np.array(image))
                image = res['image'].astype(np.float32)
                image = image.transpose(2, 0, 1)
            else:
                image = self.transform(image)

        return {'image': image, 'label': label, 'img_path': img_path}


class Five_Classes_Mass_Calc_Pathology_Dataset(Dataset):
    classes = np.array(['BACKGROUND', 'BENIGN_MASS', 'MALIGNANT_MASS', 'BENIGN_CALC', 'MALIGNANT_CALC'])

    def __init__(self, mass_root_dir, calc_root_dir, bg_root_dir, transform=None):
        self.transform = transform

        self.images_list = []
        self.labels = []

        for idx, class_name in enumerate(Mass_Calc_Pathology_Dataset.classes):
            if class_name == 'BACKGROUND':
                bg_images = glob.glob(os.path.join(bg_root_dir, '*.png'))
                self.images_list += bg_images
                self.labels += [idx] * len(bg_images)
            else:
                pathology, lesion_type = class_name.split('_')

                if lesion_type == 'MASS':
                    mass_images = glob.glob(os.path.join(
                        mass_root_dir, pathology, '*.png'))
                    self.images_list += mass_images
                    self.labels += [idx] * len(mass_images)
                elif lesion_type == 'CALC':
                    calc_images = glob.glob(os.path.join(
                        calc_root_dir, pathology, '*.png'))
                    self.images_list += calc_images
                    self.labels += [idx] * len(calc_images)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'img_path': img_path}


class  Four_Classes_Features_Pathology_Dataset(Dataset):
    classes = np.array(['BENIGN_MASS', 'MALIGNANT_MASS', 'BENIGN_CALC', 'MALIGNANT_CALC'])

    def __init__(self, mass_annotation_file, mass_root_dir, calc_annotation_file, calc_root_dir, uncertainty=0, transform=None):
        self.mass_annotations = pd.read_csv(mass_annotation_file)
        self.calc_annotations = pd.read_csv(calc_annotation_file)
        self.uncertainty = uncertainty
        self.transform = transform

        self.images_list = []
        self.labels = []
        self.lesion_types = []

        for idx, cls in enumerate(Four_Classes_Features_Pathology_Dataset.classes):
            pathology, lesion_type = cls.split("_")
            if lesion_type == "MASS":
                mass_images = glob.glob(os.path.join(
                    mass_root_dir, pathology, '*.png'))
                self.images_list += mass_images
                self.labels += [idx] * len(mass_images)
                self.lesion_types += ['MASS'] * len(mass_images)
            else:
                calc_images = glob.glob(os.path.join(
                    calc_root_dir, pathology, '*.png'))
                self.images_list += calc_images
                self.labels += [idx] * len(calc_images)
                self.lesion_types += ['CALC'] * len(calc_images)

    @staticmethod
    def convert_mass_feats_1hot(breast_density, mass_shape, mass_margins):
        BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
        BREAST_MASS_SHAPES = np.array(['ROUND', 'OVAL', 'IRREGULAR', 'LOBULATED', 'ARCHITECTURAL_DISTORTION',
                                       'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE', 'FOCAL_ASYMMETRIC_DENSITY'])
        BREAST_MASS_MARGINS = np.array(
            ['ILL_DEFINED', 'CIRCUMSCRIBED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'])

        # Breast Density
        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')

        total_ones = 0

        # Mass Shape
        if type(mass_shape) is float and math.isnan(mass_shape):
            one_hot_mass_shape = np.zeros(BREAST_MASS_SHAPES.shape, dtype=int)
        else:
            if '-' in mass_shape:
                one_hot_mass_shape = np.zeros(BREAST_MASS_SHAPES.shape, dtype=int)
                for shape in mass_shape.split('-'):
                    one_hot_mass_shape += (BREAST_MASS_SHAPES == shape).astype('int')
                    total_ones += 1
            else:
                one_hot_mass_shape = (BREAST_MASS_SHAPES == mass_shape).astype('int')
                total_ones += 1

        # Mass Margin
        if type(mass_margins) is float and math.isnan(mass_margins):
            one_hot_mass_margins = np.zeros(BREAST_MASS_MARGINS.shape, dtype=int)
        else:
            if '-' in mass_margins:
                one_hot_mass_margins = np.zeros(BREAST_MASS_MARGINS.shape, dtype=int)
                for margin in mass_margins.split('-'):
                    one_hot_mass_margins += (BREAST_MASS_MARGINS == margin).astype('int')
                    total_ones += 1
            else:
                one_hot_mass_margins = (BREAST_MASS_MARGINS ==
                                        mass_margins).astype('int')
                total_ones += 1

        ret = np.concatenate((one_hot_mass_shape, one_hot_mass_margins))
        assert np.sum(ret) == total_ones

        return ret, one_hot_breast_density 

    @staticmethod
    def convert_calc_feats_1hot(breast_density, calc_type, calc_distribution):
        if breast_density == 0:
            breast_density = 1  # one test sample is false labelling

        BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
        BREAST_CALC_TYPES = np.array(["AMORPHOUS", "PUNCTATE", "VASCULAR", "LARGE_RODLIKE", "DYSTROPHIC", "SKIN", "MILK_OF_CALCIUM",
                                      "EGGSHELL", "PLEOMORPHIC", "COARSE", "FINE_LINEAR_BRANCHING", "LUCENT_CENTER", "ROUND_AND_REGULAR", "LUCENT_CENTERED"])
        BREAST_CALC_DISTS = np.array(
            ["CLUSTERED", "LINEAR", "REGIONAL", "DIFFUSELY_SCATTERED", "SEGMENTAL"])

        # Breast Density
        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')
        total_ones = 0 

        # Calc Type
        if type(calc_type) is float and math.isnan(calc_type):
            one_hot_calc_type = np.zeros(BREAST_CALC_TYPES.shape, dtype=int)
        else:
            if '-' in calc_type:
                one_hot_calc_type = np.zeros(BREAST_CALC_TYPES.shape, dtype=int)
                for ctype in calc_type.split('-'):
                    one_hot_calc_type += (BREAST_CALC_TYPES == ctype).astype('int')
                    total_ones += 1
            else:
                one_hot_calc_type = (BREAST_CALC_TYPES == calc_type).astype('int')
                total_ones += 1

        # Calc Dist
        if type(calc_distribution) is float and math.isnan(calc_distribution):
            one_hot_calc_distribution = np.zeros(BREAST_CALC_DISTS.shape, dtype=int)
        else:
            if '-' in calc_distribution:
                one_hot_calc_distribution = np.zeros(BREAST_CALC_DISTS.shape, dtype=int)
                for dist in calc_distribution.split('-'):
                    one_hot_calc_distribution += (BREAST_CALC_DISTS == dist).astype('int')
                    total_ones += 1
            else:
                one_hot_calc_distribution = (BREAST_CALC_DISTS == calc_distribution).astype('int')
                total_ones += 1

        ret = np.concatenate(
            (one_hot_calc_type, one_hot_calc_distribution))
        assert np.sum(ret) == total_ones

        return ret, one_hot_breast_density 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        lesion_type = self.lesion_types[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        if lesion_type == 'MASS':
            roi_idx = 0
            while True:
                roi_idx += 1
                rslt_df = get_info_lesion(self.mass_annotations, f'{img_name}')

                if len(rslt_df) > 0:
                    break

            breast_density = rslt_df['breast_density'].to_numpy()[0]
            mass_shape = rslt_df['mass shape'].to_numpy()[0]
            mass_margins = rslt_df['mass margins'].to_numpy()[0]
            feature_vector, breast_density_1hot = Four_Classes_Features_Pathology_Dataset.convert_mass_feats_1hot(
                breast_density, mass_shape, mass_margins)
            feature_vector = np.concatenate((breast_density_1hot, feature_vector, np.zeros(19)))

            if random.random() < self.uncertainty:
                feature_vector = np.zeros(feature_vector.shape)
        elif lesion_type == 'CALC':
            roi_idx = 0
            while True:
                roi_idx += 1
                rslt_df = get_info_lesion(self.calc_annotations, f'{img_name}')

                if len(rslt_df) > 0:
                    break

            breast_density = rslt_df['breast density'].to_numpy()[0]
            calc_type = rslt_df['calc type'].to_numpy()[0]
            calc_distribution = rslt_df['calc distribution'].to_numpy()[0]
            feature_vector, breast_density_1hot = Four_Classes_Features_Pathology_Dataset.convert_calc_feats_1hot(
                breast_density, calc_type, calc_distribution)
            feature_vector = np.concatenate((breast_density_1hot, np.zeros(13), feature_vector))

            if random.random() < self.uncertainty:
                feature_vector = np.zeros(feature_vector.shape)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'feature_vector': feature_vector, 'img_path': img_path}


class Features_Pathology_Dataset(Dataset):
    classes = np.array(['BENIGN', 'MALIGNANT'])

    def __init__(self, lesion_type, annotation_file, root_dir, transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.root_dir = root_dir
        self.transform = transform
        self.lesion_type = lesion_type

        self.images_list = []
        self.labels = []

        for idx, pathology in enumerate(Features_Pathology_Dataset.classes):
            images = glob.glob(os.path.join(root_dir, pathology, '*.png'))
            self.images_list += images
            self.labels += [idx] * len(images)

    @staticmethod
    def convert_mass_feats_1hot(breast_density, mass_shape, mass_margins):
        BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
        BREAST_MASS_SHAPES = np.array(['ROUND', 'OVAL', 'IRREGULAR', 'LOBULATED', 'ARCHITECTURAL_DISTORTION',
                                       'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE', 'FOCAL_ASYMMETRIC_DENSITY'])
        BREAST_MASS_MARGINS = np.array(
            ['ILL_DEFINED', 'CIRCUMSCRIBED', 'SPICULATED', 'MICROLOBULATED', 'OBSCURED'])

        # Breast Density
        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')

        total_ones = 1

        # Mass Shape
        if type(mass_shape) is float and math.isnan(mass_shape):
            one_hot_mass_shape = np.zeros(BREAST_MASS_SHAPES.shape, dtype=int)
        else:
            if '-' in mass_shape:
                one_hot_mass_shape = np.zeros(BREAST_MASS_SHAPES.shape, dtype=int)
                for shape in mass_shape.split('-'):
                    one_hot_mass_shape += (BREAST_MASS_SHAPES == shape).astype('int')
                    total_ones += 1
            else:
                one_hot_mass_shape = (BREAST_MASS_SHAPES == mass_shape).astype('int')
                total_ones += 1

        # Mass Margin
        if type(mass_margins) is float and math.isnan(mass_margins):
            one_hot_mass_margins = np.zeros(BREAST_MASS_MARGINS.shape, dtype=int)
        else:
            if '-' in mass_margins:
                one_hot_mass_margins = np.zeros(BREAST_MASS_MARGINS.shape, dtype=int)
                for margin in mass_margins.split('-'):
                    one_hot_mass_margins += (BREAST_MASS_MARGINS == margin).astype('int')
                    total_ones += 1
            else:
                one_hot_mass_margins = (BREAST_MASS_MARGINS ==
                                        mass_margins).astype('int')
                total_ones += 1

        ret = np.concatenate(
            (one_hot_breast_density, one_hot_mass_shape, one_hot_mass_margins))
        assert np.sum(ret) == total_ones

        return ret

    @staticmethod
    def convert_calc_feats_1hot(breast_density, calc_type, calc_distribution):
        if breast_density == 0:
            breast_density = 1  # one test sample is false labelling

        BREAST_DENSITY_TYPES = np.array([1, 2, 3, 4])
        BREAST_CALC_TYPES = np.array(["AMORPHOUS", "PUNCTATE", "VASCULAR", "LARGE_RODLIKE", "DYSTROPHIC", "SKIN", "MILK_OF_CALCIUM",
                                      "EGGSHELL", "PLEOMORPHIC", "COARSE", "FINE_LINEAR_BRANCHING", "LUCENT_CENTER", "ROUND_AND_REGULAR", "LUCENT_CENTERED"])
        BREAST_CALC_DISTS = np.array(
            ["CLUSTERED", "LINEAR", "REGIONAL", "DIFFUSELY_SCATTERED", "SEGMENTAL"])

        # Breast Density
        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')
        total_ones = 1

        # Calc Type
        if type(calc_type) is float and math.isnan(calc_type):
            one_hot_calc_type = np.zeros(BREAST_CALC_TYPES.shape, dtype=int)
        else:
            if '-' in calc_type:
                one_hot_calc_type = np.zeros(BREAST_CALC_TYPES.shape, dtype=int)
                for ctype in calc_type.split('-'):
                    one_hot_calc_type += (BREAST_CALC_TYPES == ctype).astype('int')
                    total_ones += 1
            else:
                one_hot_calc_type = (BREAST_CALC_TYPES == calc_type).astype('int')
                total_ones += 1

        # Calc Dist
        if type(calc_distribution) is float and math.isnan(calc_distribution):
            one_hot_calc_distribution = np.zeros(BREAST_CALC_DISTS.shape, dtype=int)
        else:
            if '-' in calc_distribution:
                one_hot_calc_distribution = np.zeros(BREAST_CALC_DISTS.shape, dtype=int)
                for dist in calc_distribution.split('-'):
                    one_hot_calc_distribution += (BREAST_CALC_DISTS == dist).astype('int')
                    total_ones += 1
            else:
                one_hot_calc_distribution = (BREAST_CALC_DISTS == calc_distribution).astype('int')
                total_ones += 1

        ret = np.concatenate(
            (one_hot_breast_density, one_hot_calc_type, one_hot_calc_distribution))
        assert np.sum(ret) == total_ones

        return ret

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        image = Image.open(img_path)
        label = self.labels[idx]

        roi_idx = 0
        while True:
            roi_idx += 1
            rslt_df = get_info_lesion(self.annotations, f'{img_name}')

            if len(rslt_df) > 0:
                break

        if self.lesion_type == 'mass':
            breast_density = rslt_df['breast_density'].to_numpy()[0]
            mass_shape = rslt_df['mass shape'].to_numpy()[0]
            mass_margins = rslt_df['mass margins'].to_numpy()[0]
            feature_vector = Features_Pathology_Dataset.convert_mass_feats_1hot(
                breast_density, mass_shape, mass_margins)
        elif self.lesion_type == 'calc':
            breast_density = rslt_df['breast density'].to_numpy()[0]
            calc_type = rslt_df['calc type'].to_numpy()[0]
            calc_distribution = rslt_df['calc distribution'].to_numpy()[0]
            feature_vector = Features_Pathology_Dataset.convert_calc_feats_1hot(
                breast_density, calc_type, calc_distribution)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'feature_vector': feature_vector, 'img_path': img_path}
