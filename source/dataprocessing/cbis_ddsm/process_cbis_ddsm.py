import os
import glob
import numpy as np
import cv2
import mmcv
import pandas
import pandas as pd
import warnings
import math
import shutil

from config.cfg_loader import proj_paths_json
from pycocotools import mask as coco_api_mask
from sklearn.model_selection import StratifiedShuffleSplit
from utilities.detectutil import bbox_util
from PIL import Image
from scipy import stats
from natsort import natsorted


def convert_npz_to_png(data_path, preprocessing=None):
    '''
    Args:
    preprocessing - possible approaches: 1) 'mode_norm': (from Scientific Report Paper on INBreast)
    '''
    for dir in glob.glob(os.path.join(data_path, '*')):
        num_files = len(glob.glob(os.path.join(dir, '*')))
        if num_files < 2:
            print(os.path.basename(dir), num_files)
            continue

        if preprocessing is None:
            save_path = os.path.join(dir, os.path.basename(dir)+'.png')
        elif preprocessing == 'mode_norm':
            save_path = os.path.join(dir, 'mode_8000_12800_' + os.path.basename(dir)+'.png')
            
        # rgb_save_path = os.path.join(dir, 'rgb_' + os.path.basename(dir)+'.png')

        # check if image has existed
        if os.path.exists(save_path):
            continue
        # if os.path.exists(rgb_save_path):
        #     continue

        try:
            # For saving the original mammamogram
            if not os.path.exists(save_path):
                mamm_img = np.load(os.path.join(dir, "image.npz"),
                                allow_pickle=True)["image"]
                if preprocessing == 'mode_norm':
                    flat_img = mamm_img.flatten() 
                    flat_img = flat_img[flat_img != 0]
                    mode = stats.mode(flat_img).mode# ignore background value 0
                    # mamm_img = np.clip(mamm_img, mode-500, mode+800) # for INbreast from the paper
                    mamm_img = np.clip(mamm_img, mode-8000, mode+12800) # INbreast pixels value are from 0 to 4095
                                                                          # CBIS_DDSM pixels value are from 0 to 65535

                cv2.imwrite(save_path, mamm_img)

            # For saving the mammamogram which has been
            # converted to RGB image
            # if not os.path.exists(rgb_save_path):
            #     png_mamm = mmcv.imread(save_path)
            #     cv2.imwrite(rgb_save_path, png_mamm)
        except:
            print('zlip error:', dir)


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


def get_info_lesion(df, ROI_ID):
    _, _, patient_id, left_or_right, image_view, abnormality_id = ROI_ID.split(
        '_')
    rslt_df = df[(df['patient_id'] == ('P_' + patient_id)) &
                 (df['left or right breast'] == left_or_right) &
                 (df['image view'] == image_view) &
                 (df['abnormality id'] == int(abnormality_id))]

    return rslt_df


def convert_ddsm_to_coco(categories, out_file, data_root, annotation_filepath, extend_bb_ratio=None, keep_org_boxes=False, rgb_img=False, return_size_rate=1.0):
    '''
    Args:
    return_size_rate (float from 0 to 1.0): the numbers of images that you want to experiment with. This is used to plot the learning curve based on the size of data
    
    Returns:
    None
    '''
    save_path = os.path.join(data_root, out_file)
    if os.path.exists(save_path):
        warnings.warn(f"{save_path} has already existed")
        return

    images = []
    annotations = []
    obj_count = 0

    df = pandas.read_csv(annotation_filepath)

    total_imgs = len(glob.glob(os.path.join(data_root, '*')))

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        if idx > total_imgs * return_size_rate:
            break

        filename = os.path.basename(dir_path)
        if filename.split('_')[-1] not in ['CC', 'MLO']: # skip mask directories
            continue

        filename = os.path.basename(dir_path)

        img_path = glob.glob(os.path.join(dir_path, '**', '**', '000000.png'))[0]

        if not os.path.exists(img_path):
            continue
        img = mmcv.imread(img_path)
        height, width = img.shape[:2]

        if rgb_img:
            images.append(dict(
                id=idx,
                file_name=os.path.join(filename, 'rgb_' + filename+'.png'),
                height=height,
                width=width))
        else:
            images.append(dict(
                id=idx,
                file_name=os.path.join(filename, filename+'.png'),
                height=height,
                width=width))

        bboxes = []
        labels = []
        masks = []

        # for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
        for mask_path in glob.glob(dir_path + '_*'):
            roi_idx = mask_path.split('_')[-1]

            rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

            if len(rslt_df) == 0:
                print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                continue

            label = rslt_df['pathology'].to_numpy()[0]
            if label == 'MALIGNANT':
                cat_id = 0
            elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                cat_id = 1
            # else:
            #     raise ValueError(
            #         f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')


            # mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
            mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000001.png')))
            if len(mask_paths) != 0:
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000000.png')))
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_arr = cv2.imread(mask_paths[1], cv2.IMREAD_GRAYSCALE)

            if img.shape[:2] != mask_arr.shape[:2]:
                print('[+] Image and mask resolutions do not match')
                continue

            seg_poly = mask2polygon(mask_arr)
            seg_poly = [[el + 0.5 for el in poly] for poly in seg_poly]
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            if extend_bb_ratio is None:
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=cat_id,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=seg_area,
                    segmentation=seg_poly,
                    iscrowd=0)

                annotations.append(data_anno)

                obj_count += 1
            elif extend_bb_ratio is not None:
                if keep_org_boxes:
                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=cat_id,
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=seg_area,
                        segmentation=seg_poly,
                        iscrowd=0)

                    annotations.append(data_anno)

                    obj_count += 1

                ext_x_min, ext_y_min, ext_x_max, ext_y_max = \
                    bbox_util.extendBB(org_img_size=mask_arr.shape[:2], \
                                       left=x_min, \
                                       top=y_min, \
                                       right=x_max, \
                                       bottom=y_max, \
                                       ratio=extend_bb_ratio)
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=cat_id,
                    bbox=[ext_x_min, ext_y_min, ext_x_max -
                          ext_x_min, ext_y_max - ext_y_min],
                    area=seg_area,
                    segmentation=seg_poly,
                    iscrowd=0)

                annotations.append(data_anno)

                obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    mmcv.dump(coco_format_json, os.path.join(data_root, out_file))


def get_lesions_feature(feat1_root, feat2_root, feat3_root, data_root,
                        annotation_filepath, lesion_type, segmented=False,
                        add_mask_channel=False, patch_ext='center'):
    '''
    1) If lesion type is 'mass', we have 4 features: Mass Shape, Mass Margins, Breast Density Lesion-Level, Breast Density Image-Level
    2) Else if lesion type is 'calcification', we have 4 features: Calc Type, Calc Distribution, Breast Density Lesion-Level, Breast Density Image-Level
    '''

    df = pandas.read_csv(annotation_filepath)

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        if filename.split('_')[-1] not in ['CC', 'MLO']: # skip mask directories
            continue

        img_path = glob.glob(os.path.join(dir_path, '**', '**', '000000.png'))[0]
        if not os.path.exists(img_path):
            continue

        img = mmcv.imread(img_path)
        height, width = img.shape[:2]

        # for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
        for mask_path in glob.glob(dir_path + '_*'):
            roi_idx = mask_path.split('_')[-1]

            rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

            if len(rslt_df) == 0:
                print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                continue

            label = rslt_df['pathology'].to_numpy()[0]
            if label == 'MALIGNANT':
                cat_id = 0
            elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                cat_id = 1
            # else:
            #     raise ValueError(
            #         f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')


            # if roi_idx == 100:
            #     print(f'ROI features contain NA or combined type: {filename}')
            #     break

            # mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
            mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000001.png')))
            if len(mask_paths) != 0:
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000000.png')))
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_arr = cv2.imread(mask_paths[1], cv2.IMREAD_GRAYSCALE)

            if img.shape[:2] != mask_arr.shape[:2]:
                print('[+] Image and mask resolutions do not match')
                continue

            seg_poly = mask2polygon(mask_arr)
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            if patch_ext == 'center':
                patch_width = x_max - x_min + 1
                patch_height = y_max - y_min + 1

                center_x = (x_min + x_max)/2.0
                center_y = (y_min + y_max)/2.0

                patch_size = max(patch_width, patch_height)

                new_x_min = max(0, int(center_x - patch_size/2))
                new_y_min = max(0, int(center_y - patch_size/2))
                new_x_max = min(int(width), int(center_x + patch_size/2))
                new_y_max = min(int(height), int(center_y + patch_size/2))


            if segmented:
                img = cv2.bitwise_and(img, img, mask=mask_arr)

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



            if add_mask_channel:
                mask_patch = mask_arr[y_min:(y_max+1), x_min:(x_max+1)]
                lesion_patch[:,:,2] = mask_patch

            if lesion_type == 'mass':
                mass_shape = rslt_df['mass shape'].to_numpy()[0]
                # remove NAN
                if isinstance(mass_shape, str):
                    mass_shape_save_path = os.path.join(
                        feat1_root, mass_shape)
                    save_mass_shape_lesion_path = os.path.join(
                        mass_shape_save_path,  f'{filename}_{roi_idx}.png')
                    if not os.path.exists(save_mass_shape_lesion_path):
                        os.makedirs(mass_shape_save_path, exist_ok=True)
                        cv2.imwrite(save_mass_shape_lesion_path, lesion_patch)

                mass_margins = rslt_df['mass margins'].to_numpy()[0]
                # remove NAN
                if isinstance(mass_margins, str):
                    mass_margins_save_path = os.path.join(
                        feat2_root, mass_margins)

                    save_mass_margins_lesion_path = os.path.join(
                        mass_margins_save_path,  f'{filename}_{roi_idx}.png')
                    if not os.path.exists(save_mass_margins_lesion_path):
                        os.makedirs(mass_margins_save_path, exist_ok=True)
                        cv2.imwrite(save_mass_margins_lesion_path, lesion_patch)

                mass_density = rslt_df['breast_density'].to_numpy()[0]
                # remove NAN
                if isinstance(mass_density, np.int64):
                    # mass_density_save_path = os.path.join(feat3_root, str(mass_density))
                    # save_mass_density_lesion_path = os.path.join(mass_density_save_path, f'{filename}_{roi_idx}.png')
                    # if not os.path.exists(save_mass_density_lesion_path):
                    #     os.makedirs(mass_density_save_path, exist_ok=True)
                    #     cv2.imwrite(save_mass_density_lesion_path, lesion_patch)

                    # if feat4_root is not None:
                    mass_density_img_save_path = os.path.join(feat3_root, str(mass_density))
                    save_mass_density_img_path = os.path.join(mass_density_img_save_path, f'{filename}.png')
                    if not os.path.exists(save_mass_density_img_path):
                        os.makedirs(mass_density_img_save_path, exist_ok=True)
                        cv2.imwrite(save_mass_density_img_path, img)

            elif lesion_type == 'calc':
                calc_type = rslt_df['calc type'].to_numpy()[0]
                # remove NAN
                if isinstance(calc_type, str):
                    calc_type_save_path = os.path.join(
                        feat1_root, calc_type)
                    save_calc_type_lesion_path = os.path.join(
                        calc_type_save_path,  f'{filename}_{roi_idx}.png')
                    if not os.path.exists(save_calc_type_lesion_path):
                        os.makedirs(calc_type_save_path, exist_ok=True)
                        cv2.imwrite(save_calc_type_lesion_path, lesion_patch)

                calc_dist = rslt_df['calc distribution'].to_numpy()[0]
                # remove NAN
                if isinstance(calc_dist, str):
                    calc_dist_save_path = os.path.join(
                        feat2_root, calc_dist)

                    save_calc_dist_lesion_path = os.path.join(
                        calc_dist_save_path,  f'{filename}_{roi_idx}.png')
                    if not os.path.exists(save_calc_dist_lesion_path):
                        os.makedirs(calc_dist_save_path, exist_ok=True)
                        cv2.imwrite(save_calc_dist_lesion_path, lesion_patch)

                calc_density = rslt_df['breast density'].to_numpy()[0]
                # remove NAN
                if isinstance(calc_density, np.int64):
                    # calc_density_save_path = os.path.join(feat3_root, str(calc_density))

                    # save_calc_density_lesion_path = os.path.join(calc_density_save_path, f'{filename}_{roi_idx}.png')
                    # if not os.path.exists(save_calc_density_lesion_path):
                    #     os.makedirs(calc_density_save_path, exist_ok=True)
                    #     cv2.imwrite(save_calc_density_lesion_path, lesion_patch)

                    # if feat4_root is not None:
                    calc_density_img_save_path = os.path.join(feat3_root, str(calc_density))
                    save_calc_density_img_path = os.path.join(calc_density_img_save_path, f'{filename}.png')
                    if not os.path.exists(save_calc_density_img_path):
                        os.makedirs(calc_density_img_save_path, exist_ok=True)
                        cv2.imwrite(save_calc_density_img_path, img)


def stoa_get_lesions_pathology(save_root, data_root, annotation_filename, lesion_type, new_size=(896, 1152), patch_size=224):
    df = pandas.read_csv(os.path.join(data_root, annotation_filename))

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        img_path = os.path.join(dir_path, filename + '.png')
        if not os.path.exists(img_path):
            continue

        img = mmcv.imread(img_path)
        old_height, old_width = img.shape[:2]
        if new_size is not None:
            img = cv2.resize(img, (new_size[0], new_size[1]))
        new_height, new_width = img.shape[:2]

        for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
            while True:
                if roi_idx == 100: # assume no mammamogram contains at most 100
                    break
                roi_idx += 1

                rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

                if len(rslt_df) == 0:
                    print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                    continue

                label = rslt_df['pathology'].to_numpy()[0]
                if label == 'MALIGNANT':
                    cat_id = 0
                elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                    cat_id = 1
                else:
                    raise ValueError(
                        f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')

                break

            if roi_idx == 100:
                print(f'ROI features contain NA or combined type: {filename}')
                break

            mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
            seg_poly = mask2polygon(mask_arr)
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))
            patch_width = x_max - x_min + 1
            patch_height = y_max - y_min + 1

            center_x = (x_min + x_max)/2.0
            center_y = (y_min + y_max)/2.0

            new_center_x = (center_x/old_width)*new_width
            new_center_y = (center_y/old_height)*new_height
            if patch_size is not None:
                new_patch_size = patch_size
            else:
                new_patch_size = max(patch_width, patch_height) # set new_patch_size to fixed value 224 to match scientific report paper

            new_x_min = max(0, int(new_center_x - new_patch_size/2))
            new_y_min = max(0, int(new_center_y - new_patch_size/2))
            new_x_max = min(int(new_width), int(new_center_x + new_patch_size/2))
            new_y_max = min(int(new_height), int(new_center_y + new_patch_size/2))

            lesion_patch = img[new_y_min:(new_y_max+1), new_x_min:(new_x_max+1)]

            pad_width_size = patch_size - (new_x_max - new_x_min + 1)
            pad_height_size = patch_size - (new_y_max - new_y_min + 1)
            lesion_patch = np.pad(lesion_patch,
                                  [(pad_height_size//2, pad_height_size - pad_height_size//2),
                                   (pad_width_size//2, pad_width_size - pad_width_size//2)], 'constant')

            if cat_id == 0:
                save_path = os.path.join(save_root, 'MALIGNANT')
            elif cat_id == 1:
                save_path = os.path.join(save_root, 'BENIGN')

            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f'{filename}_{roi_idx}.png'), lesion_patch)


def get_lesions_pathology(save_root, data_root, annotation_filepath, lesion_type,
                          histeq=False, equalization_type='he', patch_ext='center',
                          birads34_only=False):
    df = pandas.read_csv(annotation_filepath)

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        if filename.split('_')[-1] not in ['CC', 'MLO']: # skip mask directories
            continue
        img_path = glob.glob(os.path.join(dir_path, '**', '**', '000000.png'))[0]
        if not os.path.exists(img_path):
            continue

        img = mmcv.imread(img_path)
        height, width = img.shape[:2]

        if histeq:
            if equalization_type == 'he':
                img = mmcv.image.photometric.imequalize(img)
            elif equalization_type == 'clahe':
                # Currently getting error due to
                # "assert img.ndim == 2"
                img = mmcv.image.photometric.clahe(img)

        # for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
        for mask_path in glob.glob(dir_path + '_*'):
            roi_idx = mask_path.split('_')[-1]

            rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

            if len(rslt_df) == 0:
                print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                continue

            label = rslt_df['pathology'].to_numpy()[0]
            if label == 'MALIGNANT':
                cat_id = 0
            elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                cat_id = 1

            if birads34_only:
                birad = rslt_df['assessment'].to_numpy()[0]

                if birad not in [3, 4]:
                    continue
            # else:
            #     raise ValueError(
            #         f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')

            # if roi_idx == 100:
            #     print(f'ROI features contain NA or combined type: {filename}')
            #     break

            # mask_arr = np.load(mask_path, allow_pickle=True)["mask"]

            mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000001.png')))
            if len(mask_paths) != 0:
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_paths = glob.glob((os.path.join(mask_path, '**', '**', '000000.png')))
                mask_arr = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

            if len(np.unique(mask_arr)) != 2:
                mask_arr = cv2.imread(mask_paths[1], cv2.IMREAD_GRAYSCALE)

            if img.shape[:2] != mask_arr.shape[:2]:
                print('[+] Image and mask resolutions do not match')
                continue

            seg_poly = mask2polygon(mask_arr)
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            if patch_ext == 'center':
                patch_width = x_max - x_min + 1
                patch_height = y_max - y_min + 1

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

            if cat_id == 0:
                save_path = os.path.join(save_root, 'MALIGNANT')
            elif cat_id == 1:
                save_path = os.path.join(save_root, 'BENIGN')

            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f'{filename}_{roi_idx}.png'), lesion_patch)


def read_annotation_json(json_file):
    data = mmcv.load(json_file)

    categories = data['categories']

    img_annotations = []

    for img_data in data['images']:
        img_id = img_data['id']

        annotations = [
            annotation for annotation in data['annotations'] if annotation['id'] == img_id]

        img_annotations.append((img_data, annotations))

    return img_annotations, categories


def save_detection_gt_for_eval(data_root, detection_gt_root):
    img_annotations, categories = read_annotation_json(json_file=os.path.join(
        data_root, 'annotation_coco_with_classes.json'))

    print(len(img_annotations))
    for img, anns in mmcv.track_iter_progress(img_annotations):
        print(img['file_name'], len(anns))
        img_filename, _ = os.path.splitext(os.path.basename(img['file_name']))

        save_path = os.path.join(
            detection_gt_root, f'{img_filename}.txt')
        if os.path.exists(save_path):
            continue
        with open(save_path, 'w') as f:
            for ann in anns:
                x, y, w, h = (str(el) for el in ann['bbox'])
                c = [el['name']
                     for el in categories if el['id'] == ann['category_id']][0]
                f.write(' '.join((c, x, y, w, h, '\n')))


def cbis_ddsm_statistic(mass_root, calc_root):
    number_mass_train_images = len(
        glob.glob(os.path.join(mass_root, 'train', 'Mass-Training*')))
    number_mass_test_images = len(
        glob.glob(os.path.join(mass_root, 'test', 'Mass-Test*')))
    number_calc_train_images = len(
        glob.glob(os.path.join(calc_root, 'train', 'Calc-Training*')))
    number_calc_test_images = len(
        glob.glob(os.path.join(calc_root, 'test', 'Calc-Test*')))

    total_mass_images = number_mass_train_images + number_mass_test_images
    total_calc_images = number_calc_train_images + number_calc_test_images

    print('*'*50)
    string = 'Processed CBIS-DDSM stats'
    print('*'*((50 - len(string))//2), string, '*'*((50-len(string))//2))
    print('*'*50)

    print('Total numbers of mass images:', total_mass_images)
    print('\tMass training images:', number_mass_train_images)
    print('\tMass test images:', number_mass_test_images)

    print('Total numbers of calc images:', total_calc_images)
    print('\tCalcification training images:', number_calc_train_images)
    print('\tCalcification test images:', number_calc_test_images)
    print()

    feature_count = dict()
    for feature in glob.glob(os.path.join(mass_root, 'cls', '*')):
        feature_name = os.path.basename(feature)
        if feature_name not in feature_count:
            feature_count[feature_name] = dict()

        for fold_idx, fold in enumerate(['train', 'val']):
            for feat_cls in glob.glob(os.path.join(feature, fold, '*')):
                feat_cls_name = os.path.basename(feat_cls)
                if feat_cls_name not in feature_count[feature_name]:
                    feature_count[feature_name][feat_cls_name] = [0, 0, 0]

                feature_count[feature_name][feat_cls_name][fold_idx] += len(
                    glob.glob(os.path.join(feat_cls, '*.png')))

    for feature_name, feature_clss in feature_count.items():
        print('*'*25, 'Feature name:', feature_name, '*'*25)
        for feat_cls, cnt in feature_clss.items():
            print('%40s %s' % (feat_cls, ' '.join('%03s' % i for i in cnt)))


def split_train_val(train_save_root, val_save_root, categories, data_root, annotation_filepath, val_ratio):
    df = pd.read_csv(annotation_filepath)

    split_info = dict()
    split_info['img_name'] = []
    split_info['pathology'] = []

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)

        if filename.split('_')[-1] not in ['CC', 'MLO']: # skip mask directories
            continue

        # img_path = os.path.join(dir_path, filename + '.png')
        # if not os.path.exists(img_path):
        #     continue

        # for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):

        all_cat_ids = []
        for mask_path in glob.glob(dir_path + '_*'):
            roi_idx = mask_path.split('_')[-1]

            rslt_df = get_info_lesion(df, f'{filename}_{roi_idx}')

            if len(rslt_df) == 0:
                print(f'No ROI was found for ROI_ID: {filename}_{roi_idx}')
                continue

            label = rslt_df['pathology'].to_numpy()[0]
            if label == 'MALIGNANT':
                cat_id = 0
            elif label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                cat_id = 1
            # else:
            #     raise ValueError(
            #         f'Label: {label} is unrecognized for ROI_ID: {filename}_{roi_idx}')

            all_cat_ids.append(cat_id)

        if len(np.unique(np.array(all_cat_ids))) != 1:
            # if different lesions have different pathologies
            cat_id = 0 # a whole mamm should be considered malignant


        for cat in categories:
            if cat['id'] == cat_id:
                cat_name = cat['name']
        split_info['img_name'].append(filename)
        split_info['pathology'].append(cat_name)

    split_info_df = pd.DataFrame(split_info)

    split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    for train_index, test_index in split.split(split_info_df, split_info_df['pathology']):
        strat_train_set = split_info_df.reindex(index=train_index)
        strat_test_set = split_info_df.reindex(index=test_index)

    print('Pathology Original Train Ratio:')
    print(split_info_df['pathology'].value_counts()/len(split_info_df))
    print('Pathology Split Train Ratio:')
    print(strat_train_set['pathology'].value_counts()/len(strat_train_set))
    print('Pathology Split Validation Ratio:')
    print(strat_test_set['pathology'].value_counts()/len(strat_test_set))

    for dirname in strat_train_set['img_name']:
        source = os.path.join(data_root, dirname)
        destination = os.path.join(train_save_root, dirname)

        if not os.path.exists(destination):
            shutil.copytree(source, destination)

        for mask_source in glob.glob(source + '_*'):
            mask_dirname = os.path.basename(mask_source)
            mask_destination = os.path.join(train_save_root, mask_dirname)

            if not os.path.exists(mask_destination):
                shutil.copytree(mask_source, mask_destination)

    for dirname in strat_test_set['img_name']:
        source = os.path.join(data_root, dirname)
        destination = os.path.join(val_save_root, dirname)

        if not os.path.exists(destination):
            shutil.copytree(source, destination)

        for mask_source in glob.glob(source + '_*'):
            mask_dirname = os.path.basename(mask_source)
            mask_destination = os.path.join(val_save_root, mask_dirname)

            if not os.path.exists(mask_destination):
                shutil.copytree(mask_source, mask_destination)


if __name__ == '__main__':
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    ################################################################
    ######################## MASS LESIONS ##########################
    ################################################################

    ################# Process Mass Lesions Mammogram ################
    mass_train_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'train')
    mass_test_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'test')
    mass_train_train_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'train_train')
    mass_train_val_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'train_val')

    # Deprecated: All original data have been converted to .png files
    # convert_npz_to_png(data_path=mass_train_root)
    # convert_npz_to_png(data_path=mass_test_root)
    # convert_npz_to_png(data_path=mass_test_root, preprocessing='mode_norm')

    categories = [{'id': 0, 'name': 'malignant-mass', 'supercategory': 'mass'},
                  {'id': 1, 'name': 'benign-mass', 'supercategory': 'mass'}]

    # # Split train val
    # split_train_val(train_save_root=mass_train_train_root, \
    #                 val_save_root=mass_train_val_root, \
    #                 categories=categories, \
    #                 data_root=mass_train_root, \
    #                 annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                  'mass_case_description_train_set.csv'),
    #                 val_ratio=0.3)

    # # Default bbox size
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=mass_train_root,
    #                      annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                       'mass_case_description_train_set.csv'))

    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=mass_test_root,
    #                      annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                       'mass_case_description_test_set.csv'))

    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=mass_train_train_root,
    #                      annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                       'mass_case_description_train_set.csv'))

    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=mass_train_val_root,
    #                      annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                       'mass_case_description_train_set.csv'))

    # Default bbox size (For different sizes of train_train set)
    # for size in range(0, 100, 10):
    #     size = size / 100.0
    #     convert_ddsm_to_coco(categories=categories,
    #                         out_file=f'annotation_coco_with_classes_sizerate={size}.json',
    #                         data_root=mass_train_train_root,
    #                         annotation_filename='mass_case_description_train_set.csv',
    #                         return_size_rate=size)


    ############## Extract Lesion Patches ##############
    # split lesion patches based on: mass shape, mass margins, breast density
    mass_shape_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_shape'])
    mass_margins_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_margins'])
    mass_breast_density_image_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_breast_density_image'])

    # for split in ['train', 'val', 'test']:
    #     if split == 'train':
    #         data_root = mass_train_train_root
    #         annotation_filename = 'mass_case_description_train_set.csv'
    #     elif split == 'val':
    #         data_root = mass_train_val_root
    #         annotation_filename = 'mass_case_description_train_set.csv'
    #     elif split == 'test':
    #         data_root = mass_test_root
    #         annotation_filename = 'mass_case_description_test_set.csv'

    #     get_lesions_feature(feat1_root=os.path.join(mass_shape_root, split),
    #                         feat2_root=os.path.join(mass_margins_root, split),
    #                         feat3_root=os.path.join(mass_breast_density_image_root, split),
    #                         data_root=data_root,
    #                         annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                          annotation_filename),
    #                         lesion_type='mass',
    #                         patch_ext='center')


    # # split lesion patches based on pathology
    # mass_pathology_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])

    # get_lesions_pathology(os.path.join(mass_pathology_root, 'train'),
    #                       data_root=mass_train_train_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_train_set.csv'),
    #                       lesion_type='mass', patch_ext='center')
    # get_lesions_pathology(os.path.join(mass_pathology_root, 'val'),
    #                       data_root=mass_train_val_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_train_set.csv'),
    #                       lesion_type='mass', patch_ext='center')
    # get_lesions_pathology(os.path.join(mass_pathology_root, 'test'),
    #                       data_root=mass_test_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_test_set.csv'),
    #                       lesion_type='mass', patch_ext='center')

    # # split lesion patches based on pathology (birads 3&4 only for val/test set)
    # mass_pathology_birads34_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_birads34_valtest'])

    # get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'train'),
    #                       data_root=mass_train_train_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_train_set.csv'),
    #                       lesion_type='mass', patch_ext='center')
    # get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'val'),
    #                       data_root=mass_train_val_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_train_set.csv'),
    #                       lesion_type='mass', patch_ext='center', birads34_only=True)
    # get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'test'),
    #                       data_root=mass_test_root,
    #                       annotation_filepath=os.path.join(processed_cbis_ddsm_root,
    #                                                        'mass_case_description_test_set.csv'),
    #                       lesion_type='mass', patch_ext='center', birads34_only=True)

    # split lesion patches based on pathology (birads 3&4 only for train/val/test)
    mass_pathology_birads34_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_birads34'])

    get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'train'),
                          data_root=mass_train_train_root,
                          annotation_filepath=os.path.join(processed_cbis_ddsm_root,
                                                           'mass_case_description_train_set.csv'),
                          lesion_type='mass', patch_ext='center', birads34_only=True)
    get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'val'),
                          data_root=mass_train_val_root,
                          annotation_filepath=os.path.join(processed_cbis_ddsm_root,
                                                           'mass_case_description_train_set.csv'),
                          lesion_type='mass', patch_ext='center', birads34_only=True)
    get_lesions_pathology(os.path.join(mass_pathology_birads34_root, 'test'),
                          data_root=mass_test_root,
                          annotation_filepath=os.path.join(processed_cbis_ddsm_root,
                                                           'mass_case_description_test_set.csv'),
                          lesion_type='mass', patch_ext='center', birads34_only=True)

    # stoa mass pathology
    # stoa_mass_pathology_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['stoa_mass_pathology'])
    # stoa_get_lesions_pathology(os.path.join(stoa_mass_pathology_root, 'train'), data_root=mass_train_train_root, annotation_filename='mass_case_description_train_set.csv', lesion_type='mass')
    # stoa_get_lesions_pathology(os.path.join(stoa_mass_pathology_root, 'val'), data_root=mass_train_val_root, annotation_filename='mass_case_description_train_set.csv', lesion_type='mass')
    # stoa_get_lesions_pathology(os.path.join(stoa_mass_pathology_root, 'test'), data_root=mass_test_root, annotation_filename='mass_case_description_test_set.csv', lesion_type='mass')

    ################################################################
    ##################### CALCIFICATION LESIONS ####################
    ################################################################

    ################# Process Calcification Lesions Mammogram ################
    calc_train_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'train')
    calc_test_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'test')
    calc_train_train_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'train_train')
    calc_train_val_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'train_val')

    calc_desc_train_path = os.path.join(processed_cbis_ddsm_root,
                                        'calc_case_description_train_set.csv')
    calc_desc_test_path = os.path.join(processed_cbis_ddsm_root,
                                       'calc_case_description_test_set.csv')

    # Deprecated: All original data have been converted to .png files
    # convert_npz_to_png(data_path=calc_train_root)
    # convert_npz_to_png(data_path=calc_test_root)

    categories = [{'id': 0, 'name': 'malignant-calc', 'supercategory': 'calcification'}, {'id': 1, 'name': 'benign-calc', 'supercategory': 'calcification'}]

    # Split train-val
    split_train_val(train_save_root=calc_train_train_root, \
                    val_save_root=calc_train_val_root, \
                    categories=categories, \
                    data_root=calc_train_root, \
                    annotation_filepath=calc_desc_train_path,
                    val_ratio=0.3)


    # Default bbox size
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=calc_train_root,
    #                      annotation_filepath=calc_desc_train_path)
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=calc_test_root,
    #                      annotation_filepath=calc_desc_test_path)
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=calc_train_train_root,
    #                      annotation_filepath=calc_desc_train_path)
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes.json',
    #                      data_root=calc_train_val_root,
    #                      annotation_filepath=calc_desc_train_path)


    # Default bbox size (For different sizes of train_train set)
    # for size in range(0, 100, 10):
    #     size = size / 100.0
    #     convert_ddsm_to_coco(categories=categories,
    #                         out_file=f'annotation_coco_with_classes_sizerate={size}.json',
    #                         data_root=calc_train_train_root,
    #                         annotation_filename='calc_case_description_train_set.csv',
    #                         return_size_rate=size)

    


    ############ EXTRACT LESION PATCHES ##############
    # split lesion patches based on: mass shape, mass margins, breast density
    calc_type_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_type'])
    calc_dist_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_dist'])
    calc_breast_density_image_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_breast_density_image'])

    # for split in ['train', 'val', 'test']:
    #     if split == 'train':
    #         data_root = calc_train_train_root
    #         annotation_filepath = calc_desc_train_path
    #     elif split == 'val':
    #         data_root = calc_train_val_root
    #         annotation_filepath = calc_desc_train_path
    #     elif split == 'test':
    #         data_root = calc_test_root
    #         annotation_filepath = calc_desc_test_path
    #     get_lesions_feature(feat1_root=os.path.join(calc_type_root, split),
    #                         feat2_root=os.path.join(calc_dist_root, split),
    #                         feat3_root=os.path.join(calc_breast_density_image_root, split),
    #                         data_root=data_root,
    #                         annotation_filepath=annotation_filepath,
    #                         lesion_type='calc',
    #                         patch_ext='center')


    # # split lesion patches based on pathology
    # calc_pathology_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

    # get_lesions_pathology(os.path.join(calc_pathology_root, 'train'), data_root=calc_train_train_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center')
    # get_lesions_pathology(os.path.join(calc_pathology_root, 'val'), data_root=calc_train_val_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center')
    # get_lesions_pathology(os.path.join(calc_pathology_root, 'test'), data_root=calc_test_root, annotation_filepath=calc_desc_test_path, lesion_type='calc', patch_ext='center')

    # split lesion patches based on pathology (birads 3&4 only for val/test set)
    # calc_pathology_birads34_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_birads34_valtest'])

    # get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'train'), data_root=calc_train_train_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center')
    # get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'val'), data_root=calc_train_val_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center', birads34_only=True)
    # get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'test'), data_root=calc_test_root, annotation_filepath=calc_desc_test_path, lesion_type='calc', patch_ext='center', birads34_only=True)

    # split lesion patches based on pathology (birads 3&4 only for train/val/test)
    calc_pathology_birads34_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_birads34'])

    get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'train'), data_root=calc_train_train_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center', birads34_only=True)
    get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'val'), data_root=calc_train_val_root, annotation_filepath=calc_desc_train_path, lesion_type='calc', patch_ext='center', birads34_only=True)
    get_lesions_pathology(os.path.join(calc_pathology_birads34_root, 'test'), data_root=calc_test_root, annotation_filepath=calc_desc_test_path, lesion_type='calc', patch_ext='center', birads34_only=True)

    # stoa_calc_pathology_root = os.path.join(processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['stoa_calc_pathology'])
    # stoa_get_lesions_pathology(os.path.join(stoa_calc_pathology_root, 'train'), data_root=calc_train_train_root, annotation_filename='calc_case_description_train_set.csv', lesion_type='calc')
    # stoa_get_lesions_pathology(os.path.join(stoa_calc_pathology_root, 'val'), data_root=calc_train_val_root, annotation_filename='calc_case_description_train_set.csv', lesion_type='calc')
    # stoa_get_lesions_pathology(os.path.join(stoa_calc_pathology_root, 'test'), data_root=calc_test_root, annotation_filename='calc_case_description_test_set.csv', lesion_type='calc')


    ############# Save detection ground-truth for evaluation ##############
    ### using https://github.com/rafaelpadilla/Object-Detection-Metrics ###
    #######################################################################
    # experiment_root = proj_paths_json['EXPERIMENT']['root']
    # processed_cbis_ddsm_detection_gt_root = os.path.join(
    #     experiment_root,
    #     proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['root'],
    #     proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['det_gt'])
    # train_det_gt_root = os.path.join(
    #     processed_cbis_ddsm_detection_gt_root, 'train')
    # test_det_gt_root = os.path.join(
    #     processed_cbis_ddsm_detection_gt_root, 'test')
    # if not os.path.exists(train_det_gt_root):
    #     os.makedirs(train_det_gt_root, exist_ok=True)
    # if not os.path.exists(test_det_gt_root):
    #     os.makedirs(test_det_gt_root, exist_ok=True)

    # save_detection_gt_for_eval(
    #     data_root=mass_train_root, detection_gt_root=train_det_gt_root)
    # save_detection_gt_for_eval(
    #     data_root=mass_test_root, detection_gt_root=test_det_gt_root)

    # Statistics of CBIS-DDSM
    cbis_ddsm_statistic(mass_root=os.path.join(processed_cbis_ddsm_root, 'mass'),
                        calc_root=os.path.join(processed_cbis_ddsm_root, 'calc'))

