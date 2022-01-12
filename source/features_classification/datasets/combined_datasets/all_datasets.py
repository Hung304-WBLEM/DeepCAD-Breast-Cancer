import numpy as np
import torch
import os
import cv2
import pandas as pd
import glob
import albumentations
import random

from torch.utils.data import Dataset
from PIL import Image
from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Five_Classes_Mass_Calc_Pathology_Dataset
from features_classification.datasets.inbreast.inbreast_datasets import INBreast_Pathology_Dataset
from features_classification.datasets.bcdr.bcdr_datasets import All_BCDR_Pathology_Dataset
from features_classification.datasets.csaw_s.csaws_datasets import CSAWS_Dataset


class All_Pathology_Datasets(Dataset):
    classes = np.array(
        list(map(lambda x: 'DDSM_' + x, Five_Classes_Mass_Calc_Pathology_Dataset.classes.tolist())) \
        + list(map(lambda x: 'INbreast_' + x, INBreast_Pathology_Dataset.classes.tolist())) \
        + list(map(lambda x: 'BCDR_' + x, All_BCDR_Pathology_Dataset.classes.tolist())) \
        + list(map(lambda x: 'CSAWS_' + x, CSAWS_Dataset.classes.tolist()))
   )

    def __init__(self,
                 ddsm_mass_root, ddsm_calc_root, ddsm_bg_root,

                 inbreast_mass_root, inbreast_calc_root, inbreast_spiculated_root,
                 inbreast_asymetry_root, inbreast_distortion_root,
                 inbreast_cluster_root, inbreast_bg_root,

                 bcdr_film_root, bcdr_digital_root, bcdr_data_type,

                 csaws_cancer_root, csaws_calc_root, csaws_axillary_root,
                 csaws_bg_root,

                 transform = None
                 ):

        self.transform = transform

        cbis_ddsm_dataset = Five_Classes_Mass_Calc_Pathology_Dataset(ddsm_mass_root,
                                                                     ddsm_calc_root,
                                                                     ddsm_bg_root,
                                                                     transform=transform)
        inbreast_dataset = INBreast_Pathology_Dataset(inbreast_mass_root,
                                                      inbreast_calc_root,
                                                      inbreast_spiculated_root,
                                                      inbreast_asymetry_root,
                                                      inbreast_distortion_root,
                                                      inbreast_cluster_root,
                                                      inbreast_bg_root,
                                                      transform=transform)
        bcdr_dataset = All_BCDR_Pathology_Dataset(bcdr_film_root,
                                                  bcdr_digital_root,
                                                  bcdr_data_type,
                                                  transform=transform
                                                  )
        csaws_dataset = CSAWS_Dataset(csaws_cancer_root,
                                      csaws_calc_root,
                                      csaws_axillary_root,
                                      csaws_bg_root,
                                      transform=transform)

        self.images_list = \
            cbis_ddsm_dataset.get_images_list() + \
            inbreast_dataset.get_images_list() + \
            bcdr_dataset.get_images_list() + \
            csaws_dataset.get_images_list()

        ddsm_num_classes = len(Five_Classes_Mass_Calc_Pathology_Dataset.classes)
        inbreast_num_classes = len(INBreast_Pathology_Dataset.classes)
        bcdr_num_classes = len(All_BCDR_Pathology_Dataset.classes)
        csaws_num_classes = len(CSAWS_Dataset.classes)

        self.labels = \
            cbis_ddsm_dataset.get_labels() + \
            list(map(lambda x: x + ddsm_num_classes, inbreast_dataset.get_labels())) + \
            list(map(lambda x: x + ddsm_num_classes + inbreast_num_classes, bcdr_dataset.get_labels())) + \
            list(map(lambda x: x + ddsm_num_classes +
                     inbreast_num_classes + bcdr_num_classes, csaws_dataset.get_labels()))

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
        
