import os
import glob
import pandas as pd
import mmcv
import cv2

from dataprocessing.random_patches_sampling import _sample_negative_patches
from config.cfg_loader import proj_paths_json
from dataprocessing.cbis_ddsm.remove_blank_background import remove_background_images


def get_csawm_background_crops(data_root, save_root, split):
    '''
    split - either 'train' or 'test'
    '''

    csv = pd.read_csv(os.path.join(data_root, 'labels', f'CSAW-M_{split}.csv'),
                      delimiter=';')

    abnormal_masks = []
    abnormal_areas = []

    for index, row in csv.iterrows():
        filename = row['Filename']
        if_cancer = row['If_cancer']

        if if_cancer == 0:
            label = 'BENIGN'
        elif if_cancer == 1:
            label = 'MALIGNANT'
        else:
            raise ValueError

        file_path = os.path.join(data_root, 'images', 'preprocessed', split, filename)

        mamm_img = mmcv.imread(file_path)
        mamm_img = cv2.resize(mamm_img, (896, 1152))

        save_filename, _ = os.path.splitext(filename)

        for patch_id, neg_patch in enumerate(_sample_negative_patches(mamm_img,
                                                                      abnormal_masks,
                                                                      abnormal_areas,
                                                                      (224, 224))):
            save_path = os.path.join(save_root, label,
                                     save_filename.strip() + f'_background_{patch_id}.png')
            print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cv2.imwrite(save_path, neg_patch)


if __name__ == '__main__':
    data_root = proj_paths_json['DATA']['root']
    csawm = proj_paths_json['DATA']['CSAW-M']
    csawm_root = os.path.join(data_root, csawm['root'])

    bg_save_root = csawm['background']['bg_tfds']

    # Train Split
    get_csawm_background_crops(csawm_root,
                               os.path.join(csawm_root, os.path.join(bg_save_root, 'train')),
                               split='train')
    remove_background_images(os.path.join(csawm_root,
                                          os.path.join(bg_save_root, 'train', 'BENIGN')))
    remove_background_images(os.path.join(csawm_root,
                                          os.path.join(bg_save_root, 'train', 'MALIGNANT')))

    # Test Split
    get_csawm_background_crops(csawm_root,
                               os.path.join(csawm_root, os.path.join(bg_save_root, 'test')),
                               split='test')
    remove_background_images(os.path.join(csawm_root,
                                          os.path.join(bg_save_root, 'test', 'BENIGN')))
    remove_background_images(os.path.join(csawm_root,
                                          os.path.join(bg_save_root, 'test', 'MALIGNANT')))
