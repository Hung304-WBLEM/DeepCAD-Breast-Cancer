import os
import glob
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from dataprocessing.process_cbis_ddsm import get_info_lesion


class Mass_Shape_Dataset(Dataset):
    classes = np.array(['ROUND', 'OVAL', 'IRREGULAR',
                        'LOBULATED', 'ARCHITECTURAL_DISTORTION',
                        'ASYMMETRIC_BREAST_TISSUE', 'LYMPH_NODE',
                        'FOCAL_ASYMMETRIC_DENSITY'])

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

        if self.transform:
            image = self.transform(image)

        return image, label


class Mass_Margins_Dataset(Dataset):
    classes = np.array(['ILL_DEFINED',
                        'CIRCUMSCRIBED',
                        'SPICULATED',
                        'MICROLOBULATED',
                        'OBSCURED'])

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

        if self.transform:
            image = self.transform(image)

        return image, label


class Calc_Type_Dataset(Dataset):
    classes = np.array(["AMORPHOUS", "PUNCTATE", "VASCULAR",
                        "LARGE_RODLIKE", "DYSTROPHIC", "SKIN",
                        "MILK_OF_CALCIUM", "EGGSHELL", "PLEOMORPHIC",
                        "COARSE", "FINE_LINEAR_BRANCHING",
                        "LUCENT_CENTER", "ROUND_AND_REGULAR",
                        "LUCENT_CENTERED"])

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

        if self.transform:
            image = self.transform(image)

        return image, label


class Calc_Dist_Dataset(Dataset):
    classes = np.array(["CLUSTERED",
                       "LINEAR",
                       "REGIONAL",
                       "DIFFUSELY_SCATTERED",
                       "SEGMENTAL"])

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

        if self.transform:
            image = self.transform(image)

        return image, label


class Breast_Density_Dataset(Dataset):
    classes = np.array([1, 2, 3, 4])

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

        return image, label


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

        return image, label


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

        return image, label


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

        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')
        one_hot_mass_shape = (BREAST_MASS_SHAPES == mass_shape).astype('int')
        one_hot_mass_margins = (BREAST_MASS_MARGINS ==
                                mass_margins).astype('int')

        ret = np.concatenate(
            (one_hot_breast_density, one_hot_mass_shape, one_hot_mass_margins))
        assert np.sum(ret) >= 3

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

        one_hot_breast_density = (
            BREAST_DENSITY_TYPES == breast_density).astype('int')
        one_hot_calc_type = (BREAST_CALC_TYPES == calc_type).astype('int')
        one_hot_calc_distribution = (
            BREAST_CALC_DISTS == calc_distribution).astype('int')

        ret = np.concatenate(
            (one_hot_breast_density, one_hot_calc_type, one_hot_calc_distribution))
        assert np.sum(ret) >= 3

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

        return {'image': image, 'pathology': label, 'feature_vector': feature_vector}
