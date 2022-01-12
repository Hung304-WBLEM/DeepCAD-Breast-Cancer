import cv2
import mmcv
import numpy as np
import os
import glob

from config.cfg_loader import proj_paths_json


def is_background(img):
    height, width = img.shape[:2]
    total_px = height * width

    total_zero_px = np.count_nonzero(img == 0)

    if total_zero_px * 1.0 / total_px > 0.9:
        return True
    return False


def remove_background_images(data_dir):
    for img_path in mmcv.track_iter_progress(glob.glob(os.path.join(data_dir, '*.png'))):
        img = mmcv.imread(img_path)

        if is_background(img):
            os.remove(img_path)


if __name__ == '__main__':
    

    data_root = proj_paths_json['DATA']['root']
    tfds_cbis_ddsm = proj_paths_json['DATA']['CBIS_DDSM_tfds']
    tfds_cbis_ddsm_root = os.path.join(
        data_root, tfds_cbis_ddsm['root'])

    train_background_root = os.path.join(tfds_cbis_ddsm_root, 'train', 'BACKGROUND')
    remove_background_images(train_background_root)

    val_background_root = os.path.join(tfds_cbis_ddsm_root, 'val', 'BACKGROUND')
    remove_background_images(val_background_root)

    test_background_root = os.path.join(tfds_cbis_ddsm_root, 'test', 'BACKGROUND')
    remove_background_images(test_background_root)
