import os
import glob
import pydicom
import mmcv
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

from config.cfg_loader import proj_paths_json
from natsort import natsorted
from scipy import stats
from pycocotools import mask as coco_api_mask
from dataprocessing.random_patches_sampling import _sample_positive_patches
from dataprocessing.random_patches_sampling import _sample_negative_patches
from dataprocessing.cbis_ddsm.remove_blank_background import remove_background_images


def convert_dicom_to_png(data_root):
    os.makedirs(os.path.join(data_root, 'AllPNGs'), exist_ok=True)

    for img_idx, dcm_path in enumerate(mmcv.track_iter_progress(natsorted(glob.glob(os.path.join(data_root, '**', '**', '**', '*.dcm'))))):
        dcm_filename, _ = os.path.splitext(os.path.basename(dcm_path))

        dirname_1 = os.path.basename(os.path.dirname(dcm_path))
        dirname_2 = os.path.basename(os.path.dirname(os.path.dirname(dcm_path)))
        dirname_3 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(dcm_path))))

        mamm_img = pydicom.dcmread(dcm_path).pixel_array

        save_path = os.path.join(data_root, 'AllPNGs', dirname_3,
                                    dirname_2, dirname_1, dcm_filename+'.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            out = np.zeros(mamm_img.shape, np.double)
            normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
            normalized = normalized.astype(np.uint8)
            cv2.imwrite(save_path, normalized)


def get_cmmd_background_crops(data_root, save_root):
    csv = pd.read_excel(os.path.join(data_root, 'CMMD_clinicaldata_revision.xlsx'))

    for index, row in csv.iterrows():
        ID = row['ID1']
        leftright = row['LeftRight']
        age = row['Age']
        number = row['number']
        abnormal = row['abnormality']
        pathology = row['classification']

        if abnormal == 'mass':
            abn = 'MASS'
        elif abnormal == 'calcification':
            abn = 'CALC'
        elif abnormal == 'both':
            abn = 'MASS_CALC'
        else:
            print(abnormal)
            raise ValueError

        if pathology == 'Benign':
            lbl = 'BENIGN'
        elif pathology == 'Malignant':
            lbl = 'MALIGNANT'
        else:
            raise ValueError

        abnormal_masks = []
        abnormal_areas = []
        for img_path in glob.glob(os.path.join(data_root, 'AllPNGs', ID, '**', '**', '*.png')):
            mamm_img = mmcv.imread(img_path)
            mamm_img = cv2.resize(mamm_img, (896, 1152))

            save_filename = ID + '_' + os.path.basename(img_path)


            for patch_id, neg_patch in enumerate(_sample_negative_patches(mamm_img,
                                                                        abnormal_masks,
                                                                        abnormal_areas,
                                                                        (224, 224))):
                save_path = os.path.join(save_root, abn, lbl,
                                        save_filename.strip() + f'_background_{patch_id}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                cv2.imwrite(save_path, neg_patch)
        


if __name__ == '__main__':
    data_root = proj_paths_json['DATA']['root']
    cmmd = proj_paths_json['DATA']['CMMD']
    cmmd_root = os.path.join(data_root, cmmd['root'])

    # convert_dicom_to_png(data_root)

    bg_save_root = cmmd['background']['bg_tfds']
    get_cmmd_background_crops(cmmd_root, os.path.join(cmmd_root, bg_save_root))

    for abn in ['MASS', 'CALC', 'MASS_CALC']:
        remove_background_images(os.path.join(cmmd_root, bg_save_root, abn, 'BENIGN'))
        remove_background_images(os.path.join(cmmd_root, bg_save_root, abn, 'MALIGNANT'))
