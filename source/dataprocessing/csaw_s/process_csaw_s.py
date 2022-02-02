import os
import numpy as np
import mmcv
import cv2
import glob

from config.cfg_loader import proj_paths_json
from pycocotools import mask as coco_api_mask
from dataprocessing.random_patches_sampling import _sample_positive_patches
from dataprocessing.random_patches_sampling import _sample_negative_patches
from dataprocessing.cbis_ddsm.remove_blank_background import remove_background_images
from dataprocessing.inbreast.process_inbreast import increase_constrast_by_clip


def mask2polygon(_mask):
    contours, hierarchy = cv2.findContours(
        _mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation


def area(_mask):
    rle = coco_api_mask.encode(np.asfortranarray(_mask))
    area = coco_api_mask.area(rle)
    return area


def get_csaws_features(data_root,
                       axillary_lymph_nodes_save_root,
                       calcifications_save_root,
                       cancer_save_root,
                       background_save_root=None,
                       patch_ext='center',
                       use_norm=False):
    for patient_dirpath in mmcv.track_iter_progress(glob.glob(os.path.join(data_root, 'CsawS', 'anonymized_dataset', '*'))):
        patient_id = os.path.basename(patient_dirpath)

        idx = 0
        while True:
            img_path = os.path.join(patient_dirpath, f'{patient_id}_{idx}.png')

            if not os.path.exists(img_path):
                break

            img = mmcv.imread(img_path)
            if use_norm:
                mamm_img = increase_constrast_by_clip(img,
                                                      lowerbound_clip_rate=0.03,
                                                      upperbound_clip_rate=0.003)

            resized_img = cv2.resize(img, (896, 1152))
            height, width = img.shape[:2]

            img_name, _ = os.path.splitext(os.path.basename(img_path))

            abnormal_masks = []
            abnormal_areas = []

            for mask in ['axillary_lymph_nodes', 'calcifications', 'cancer']:
                mask_name = img_name + '_' + mask + '.png'
                mask_path = os.path.join(patient_dirpath, mask_name)

                mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if len(np.unique(mask_arr)) == 1:
                    continue

                seg_poly = mask2polygon(mask_arr)
                seg_area = area(mask_arr)

                flat_seg_poly = [el for sublist in seg_poly for el in sublist]
                px = flat_seg_poly[::2]
                py = flat_seg_poly[1::2]
                x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

                draw_xy_seg_poly = []
                for x, y in zip(px, py):
                    draw_xy_seg_poly.append([float(x), float(y)])

                patch_width = x_max - x_min + 1
                patch_height = y_max - y_min + 1

                # Skip too small images
                if patch_width * patch_height < 32*32:
                    continue


                if patch_ext == 'random':
                    mask_img = np.zeros([height, width], dtype=np.uint8)
                    cv2.fillPoly(mask_img, [np.array(draw_xy_seg_poly, np.int32)], 255)
                    resized_mask_img = cv2.resize(mask_img, (896, 1152))

                    mask_area = np.sum(resized_mask_img > 0)

                    abnormal_masks.append(resized_mask_img)
                    abnormal_areas.append(mask_area)


                if patch_ext == 'center':
                    center_x = (x_min + x_max)/2.0
                    center_y = (y_min + y_max)/2.0

                    patch_size = max(patch_width, patch_height)

                    new_x_min = max(0, int(center_x - patch_size/2))
                    new_y_min = max(0, int(center_y - patch_size/2))
                    new_x_max = min(int(width), int(center_x + patch_size/2))
                    new_y_max = min(int(height), int(center_y + patch_size/2))

                if patch_ext == 'exact':
                    lesion_patch = img[y_min:(y_max+1), x_min:(x_max+1)]
                elif patch_ext == 'center':
                    lesion_patch = img[new_y_min:(new_y_max+1), new_x_min:(new_x_max+1)]
                    pad_width_size = patch_size - (new_x_max - new_x_min)
                    pad_height_size = patch_size - (new_y_max - new_y_min)

                    lesion_patch = np.pad(lesion_patch,
                                        [(pad_height_size//2, pad_height_size - pad_height_size//2),
                                        (pad_width_size//2, pad_width_size - pad_width_size//2),
                                        (0, 0)], 'constant')


                if mask == 'axillary_lymph_nodes':
                    save_root = axillary_lymph_nodes_save_root
                elif mask == 'calcifications':
                    save_root = calcifications_save_root
                elif mask == 'cancer':
                    save_root = cancer_save_root


                os.makedirs(save_root, exist_ok=True)

                if patch_ext in ['exact', 'center']:
                    cv2.imwrite(os.path.join(save_root, mask_name), lesion_patch)
                elif patch_ext == 'random':
                    mask_filename, _ = os.path.splitext(mask_name)
                    for patch_id, pos_patch in enumerate(_sample_positive_patches(resized_img,
                                                                                resized_mask_img,
                                                                                mask_area,
                                                                                (224, 224))):
                        cv2.imwrite(os.path.join(save_root,
                                                f'{mask_filename}_patch_{patch_id}.png'),
                                    pos_patch)

            if patch_ext == 'random':
                os.makedirs(background_save_root, exist_ok=True)
                for patch_id, neg_patch in enumerate(_sample_negative_patches(resized_img,
                                                                            abnormal_masks,
                                                                            abnormal_areas,
                                                                            (224, 224))):
                    cv2.imwrite(os.path.join(background_save_root,
                                            f'{img_name}_background_{patch_id}.png'),
                                neg_patch)


            idx += 1



if __name__ == '__main__':
    csaws = proj_paths_json['DATA']['CSAW-S']
    data_root = os.path.join(proj_paths_json['DATA']['root'], csaws['root'])

    axillary_lymph_nodes_save_root = os.path.join(data_root, csaws['norm_axillary_lymph_nodes'])
    calcifications_save_root = os.path.join(data_root, csaws['norm_calcifications'])
    cancer_save_root = os.path.join(data_root, csaws['norm_cancer'])

    get_csaws_features(data_root,
                       axillary_lymph_nodes_save_root,
                       calcifications_save_root,
                       cancer_save_root,
                       use_norm=True)

    aug_axillary_lymph_nodes_save_root = os.path.join(data_root,
                                                      csaws['norm_aug_axillary_lymph_nodes'])
    aug_calcifications_save_root = os.path.join(data_root,
                                                csaws['norm_aug_calcifications'])
    aug_cancer_save_root = os.path.join(data_root,
                                        csaws['norm_aug_cancer'])
    background_save_root = os.path.join(data_root,
                                        csaws['norm_background']['norm_bg_tfds'])
    get_csaws_features(data_root,
                       aug_axillary_lymph_nodes_save_root,
                       aug_calcifications_save_root,
                       aug_cancer_save_root,
                       background_save_root,
                       patch_ext='random',
                       use_norm=True)
    remove_background_images(background_save_root)
