import os
import glob
import cv2
import mmcv
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import random

from natsort import natsorted
from config.cfg_loader import proj_paths_json
from absl import logging


def _patch_overlaps_any_abnormality_above_threshold(y, x, patch_size,
                                                    abnormalities_masks,
                                                    abnormalities_areas,
                                                    min_overlap_threshold):
  """Return True if the given patch overlaps significantly with any abnormality.
  Given a patch and a single abnormality, the overlap between the two is
  significant if, and only if, the relative area of the intersection of the two
  w.r.t. the area of the patch is above `min_overlap_threshold` OR the
  area of the intersection w.r.t. the total abnormality area is above
  `min_overlap_threshold`.
  Args:
    y: Top-most coordinate of the patch.
    x: Left-most coordinate of the patch.
    patch_size: Tuple with (height, width) of the patch.
    abnormalities_masks: List with the binary mask of each abnormality.
    abnormalities_areas: List with the total area of each abnormality.
    min_overlap_threshold:
  Returns:
    Returns True if the above condition is met for any of the given
    abnormalities, or False otherwise.
  """
  patch_area = patch_size[0] * patch_size[1]
  for abnorm_mask, abnorm_area in zip(abnormalities_masks, abnormalities_areas):
    abnorm_in_patch_area = np.sum(
        abnorm_mask[y:(y + patch_size[0]), x:(x + patch_size[1])] > 0)
    abnorm_in_patch_wrt_patch = abnorm_in_patch_area / patch_area
    abnorm_in_patch_wrt_abnorm = abnorm_in_patch_area / abnorm_area
    if (abnorm_in_patch_wrt_patch > min_overlap_threshold or
        abnorm_in_patch_wrt_abnorm > min_overlap_threshold):
      return True
  return False


def _find_contours(*args, **kwargs):
  tuple_ = cv2.findContours(*args, **kwargs)
  if len(tuple_) == 2:  # Recent opencv returns: (contours, hierachy)
    return tuple_[0]
  elif len(tuple_) == 3:  # Old opencv returns: (ret, contours, hierachy)
    return tuple_[1]
  else:
    raise AssertionError('Unknown {}')


def _get_roi_from_mask(mask):
  contours = _find_contours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours_areas = [cv2.contourArea(cont) for cont in contours]
  biggest_contour_idx = np.argmax(contours_areas)
  return contours[biggest_contour_idx]


def _sample_positive_patches(image,
                             abnormality_mask,
                             abnormality_area,
                             patch_size,
                             number_of_patches=10,
                             min_overlap_threshold=0.90,
                             max_number_of_trials_per_threshold=100):
    """Sample random patches from the image overlapping with the given abnormality.
    The abnormal area of the patch with respect to either (a) the total area of
    the patch, or (b) the total area of the abnormality, must be at least
    `min_overlap_threshold` (i.e. 90% by default).
    After `max_number_of_trials_per_threshold` samples, if not enough patches
    meeting this requirement have been generated, the `min_overlap_threshold` is
    reduced by 5%. This procedure is repeated until min_overlap_threshold < 0.1
    (which should not happen ever, if the dataset is correct).
    Args:
    image: Image to patch from.
    abnormality_mask: Binary mask of the abnormality in the image.
    abnormality_area: Precomputed area of the abnormality.
    patch_size: Size of the patch to extract.
    number_of_patches: Number of patches to sample around the abnormality ROI.
    min_overlap_threshold: Minimum relative area of the patch overlapping with
        the abnormality.
    max_number_of_trials_per_threshold: Maximum number of random samples to try
        before reducing the `min_overlap_threshold` by 5%.
    Yields:
    The patch cropped from the input image.
    """
    # cv2 = tfds.core.lazy_imports.cv2

    # The paper trying to be reproduced states that 90% of the are of each
    # positive patch should correspond to abnormal tissue. Thus if the total area
    # of abnormality is smaller than 0.9 * patch_area, we are certain that no
    # patch can meet this requirement. This happens indeed quite often.
    #
    # However, in a piece of code release by the authors of the paper
    # (https://github.com/yuyuyu123456/CBIS-DDSM/blob/bf3abc6ac2890b9b51eb5125e00056e39295fa44/ddsm_train/sample_patches_combined.py#L26)
    # the authors accept a patch if the total area of abnormality in the patch is
    # greater than 75% OR if 75% of the total abnormal area is in the patch.
    # In addition, they reduce the overlapping threholds every 1000 trials to
    # handle some corner casses.
    seed=42
    np.random.seed(seed)
    random.seed(seed)

    abnormality_roi = _get_roi_from_mask(abnormality_mask)
    abnorm_x, abnorm_y, abnorm_w, abnorm_h = cv2.boundingRect(abnormality_roi)

    number_of_yielded_patches = 0
    while min_overlap_threshold > 0.1:
        # Determine the region where random samples should be sampled from.
        max_h, min_h = max(abnorm_h, patch_size[0]), min(abnorm_h, patch_size[0])
        max_w, min_w = max(abnorm_w, patch_size[1]), min(abnorm_w, patch_size[1])
        min_y = abnorm_y - max_h + min_overlap_threshold * min_h
        min_x = abnorm_x - max_w + min_overlap_threshold * min_w
        max_y = abnorm_y + abnorm_h - int(min_overlap_threshold * min_h)
        max_x = abnorm_x + abnorm_w - int(min_overlap_threshold * min_w)
        # Ensure that all sampled batches are within the image.
        min_y = max(min_y, 0)
        min_x = max(min_x, 0)
        max_y = max(min(max_y, image.shape[0] - patch_size[0] - 1), min_y)
        max_x = max(min(max_x, image.shape[1] - patch_size[1] - 1), min_x)

        # Cap the number of trials if the sampling region is too small.
        effective_range_size = max_number_of_trials_per_threshold
        if (max_y - min_y + 1) * (max_x - min_x + 1) < effective_range_size:
            logging.debug(
                'The sampling region for patches of size %r with '
                'min_overlap_threshold=%f contains less possible patches than '
                'max_number_of_trials_per_threshold=%d, in abnormality',
                patch_size, min_overlap_threshold, max_number_of_trials_per_threshold)
            effective_range_size = (max_y - min_y + 1) * (max_x - min_x + 1)

        for _ in range(effective_range_size):
            patch_y = np.random.randint(min_y, max_y + 1)
            patch_x = np.random.randint(min_x, max_x + 1)
            if _patch_overlaps_any_abnormality_above_threshold(
                patch_y, patch_x, patch_size, [abnormality_mask], [abnormality_area],
                min_overlap_threshold):
                number_of_yielded_patches += 1
                yield image[patch_y:(patch_y + patch_size[0]),
                            patch_x:(patch_x + patch_size[1])]
            # If we have yielded all requested patches return.
            if number_of_yielded_patches >= number_of_patches:
                return

        # We failed to produce patches with the minimum overlapping requirements.
        # Reduce those requirements and try again.
        min_overlap_threshold = min_overlap_threshold * 0.95
        logging.debug(
            'Overlapping constraints relaxed to min_overlap_threshold=%f while '
            'sampling positive patches for the abnormality',
            min_overlap_threshold)

    # This should not happen ever.
    raise ValueError(
        'Only %d positive patches of size %r could be sampled satisfying the '
        'current conditions (min. relative overlapping area = %f) for the '
        'abnormality' % (number_of_yielded_patches, patch_size,
                         min_overlap_threshold))


def _get_breast_mask(image, min_breast_color_threshold=0.05):
    """Get the binary mask of the breast region of the image."""
    threshold = int(image.max() * min_breast_color_threshold)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours = _find_contours(image_binary, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours_areas = [cv2.contourArea(cont) for cont in contours]
    biggest_contour_idx = np.argmax(contours_areas)
    return cv2.drawContours(
        np.zeros_like(image_binary), contours, biggest_contour_idx, 255,
        cv2.FILLED)


def _sample_negative_patches(image,
                             abnormalities_masks,
                             abnormalities_areas,
                             patch_size,
                             number_of_patches=10,
                             min_breast_overlap_threshold=0.75,
                             max_abnorm_overlap_threshold=0.35,
                             max_number_of_trials_per_threshold=100):
    """Sample background patches from the image.
    The relative area of breast tissue in the patch must be, at least,
    `min_breast_overlap_threshold` of the total patch area. This is to prevent
    too easy negative examples.
    Similarly, the relative area of the abnormal tissue in the patch must be,
    at most, `max_abnorm_overlap_threshold`
    The relative area of the patch must overlap with the breast tissue with,
    at least, `min_breast_overlap_threshold` (relative) pixels.
    In addition, it must also overlap with abnormal tissue with, at most,
    `max_abnorm_overlap_threshold` (relative) pixels.
    Args:
        image: Image to patch from.
        abnormalities_masks: List of binary mask of each abnormality in the image.
        abnormalities_areas: List of precomputed area of each abnormality.
        patch_size: Size of the patch to extract.
        number_of_patches: Number of negative patches to sample from the image.
        min_breast_overlap_threshold: Minimum (relative) number of breast pixels in
        the patch.
        max_abnorm_overlap_threshold: Maximum (relative) number of abnormal pixels
        in the patch.
        max_number_of_trials_per_threshold: Maximum number of random samples to try
        before reducing the `min_breast_overlap_threshold` by 5% and increasing
        the `max_abnorm_overlap_threshold` by 5%.
    Yields:
        The patch cropped from the input image.
    """
    seed=42
    np.random.seed(seed)
    random.seed(seed)


    breast_mask = _get_breast_mask(image)

    def patch_overlapping_breast_is_feasible(y, x):
        """Return True if the patch contains enough breast pixels."""
        breast_in_patch = breast_mask[y:(y + patch_size[0]), x:(x + patch_size[1])]
        return (np.sum(breast_in_patch > 0) /
                (patch_size[0] * patch_size[1]) > min_breast_overlap_threshold)

    breast_roi = _get_roi_from_mask(breast_mask)
    breast_x, breast_y, breast_w, breast_h = cv2.boundingRect(breast_roi)
    number_of_yielded_patches = 0
    while (min_breast_overlap_threshold > 0.1 and
            max_abnorm_overlap_threshold < 0.9):
        # Determine the region where random samples should be sampled from.
        max_h, min_h = max(breast_h, patch_size[0]), min(breast_h, patch_size[0])
        max_w, min_w = max(breast_w, patch_size[1]), min(breast_w, patch_size[1])
        min_y = breast_y - int((1.0 - min_breast_overlap_threshold) * max_h)
        min_x = breast_x - int((1.0 - min_breast_overlap_threshold) * max_w)
        max_y = breast_y + breast_h - int(min_breast_overlap_threshold * min_h)
        max_x = breast_x + breast_w - int(min_breast_overlap_threshold * min_w)
        # Ensure that all sampled batches are within the image.
        min_y = max(min_y, 0)
        min_x = max(min_x, 0)
        max_y = max(min(max_y, image.shape[0] - patch_size[0] - 1), min_y)
        max_x = max(min(max_x, image.shape[1] - patch_size[1] - 1), min_x)
        # Cap the number of trials if the sampling region is too small.
        effective_range_size = max_number_of_trials_per_threshold
        if (max_y - min_y + 1) * (max_x - min_x + 1) < effective_range_size:
            logging.debug(
                'The sampling region for negative patches of size %r with '
                'min_breast_overlap_threshold=%f contains less possible patches '
                'than max_number_of_trials_per_threshold=%d, in mammography',
                patch_size, min_breast_overlap_threshold,
                max_number_of_trials_per_threshold)
            effective_range_size = (max_y - min_y + 1) * (max_x - min_x + 1)
        for _ in range(effective_range_size):
            patch_y = np.random.randint(min_y, max_y + 1)
            patch_x = np.random.randint(min_x, max_x + 1)
            if (patch_overlapping_breast_is_feasible(patch_y, patch_x) and
                not _patch_overlaps_any_abnormality_above_threshold(
                    patch_y, patch_x, patch_size, abnormalities_masks,
                    abnormalities_areas, max_abnorm_overlap_threshold)):
                number_of_yielded_patches += 1
                yield image[patch_y:(patch_y + patch_size[0]),
                            patch_x:(patch_x + patch_size[1])]

            # If we have yielded all requested patches return.
            if number_of_yielded_patches >= number_of_patches:
                return
        # We failed to produce patches with the given overlapping requirements.
        # Relaxate the requirements and try again.
        min_breast_overlap_threshold = min_breast_overlap_threshold * 0.95
        max_abnorm_overlap_threshold = max_abnorm_overlap_threshold * 1.05
        logging.debug(
            'Overlapping constraints relaxed to min_breast_overlap_threshold=%f '
            'and max_abnorm_overlap_threshold=%f while sampling negative '
            'patches for the mammography', min_breast_overlap_threshold,
            max_abnorm_overlap_threshold)  # Filepath to the abnormality mask image.

    # This should not happen ever.
    raise ValueError(
        'Only %d negative patches of size %r could be sampled satisfying the '
        'current conditions (min. relative overlapping area with breast = %f, '
        'max. relative overlapping area with abnormalities = %f) for the '
        'mammography' %
        (number_of_yielded_patches, patch_size, min_breast_overlap_threshold,
        max_abnorm_overlap_threshold))


if __name__ == '__main__':
    inbreast = proj_paths_json['DATA']['INbreast']
    data_root = os.path.join(proj_paths_json['DATA']['root'], inbreast['root'])

    for img_id, img_path in enumerate(natsorted(glob.glob(os.path.join(data_root, 'AllNormPNGs', '*.png')))):
        filename, _ = os.path.splitext(os.path.basename(img_path))

        xml_path = os.path.join(data_root, 'AllXML', filename.split('_')[0] + '.xml') 

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        _, _, _, num_rois, _, rois = root.getchildren()[0].getchildren()[1].getchildren()[0].getchildren()

        img = mmcv.imread(img_path)
        height, width = img.shape[:2]
        num_rois = num_rois.text


        labels_data = pd.read_csv(os.path.join(data_root, 'INbreast.csv'), sep=';')
        label = labels_data[labels_data['File Name'] == int(filename.split('_')[0])]['Bi-Rads'].iloc[0]
        if label in ['1', '2', '3']:
            label = 'BENIGN'
            cat_id = 1
        elif label in ['4a', '4b', '4c', '5', '6']:
            label = 'MALIGNANT'
            cat_id = 0
        else:
            print(label)
            raise ValueError

        abnormal_masks = []
        abnormal_areas = []

        for roi_idx, roi in enumerate(rois.getchildren()):
            _, _area, _, center, _, dev, _, idx_in_img, _, _max, _, mean, _, _min, _, name, _, num_pts, _, pointmm, _, pointpx, _, total, _, _type = roi.getchildren()


            if name.text in ['Calcification', 'Calcifications']:
                # Skip some calcifications because some crop is too small for INBreast dataset
                # (e.g.: a small dot)
                # print(_area.text, type(_area.text))
                # if float(_area.text) == 0:
                #     continue
                # save_root = calc_save_root
                continue
            # elif name.text in ['Mass']:
            #     save_root = mass_save_root
            # elif name.text in ['Cluster']:
            #     save_root = cluster_save_root
            # elif name.text in ['Distortion']:
            #     save_root = distortion_save_root
            # elif name.text in ['Spiculated region',
            #                    'Espiculated Region', 'Spiculated Region']:
            #     save_root = spiculated_save_root
            # elif name.text in ['Asymmetry', 'Assymetry']:
            #     save_root = asymetry_save_root
            # else:
            #     continue

            seg_area = _area.text
            coords = pointpx.getchildren()
            xy_seg_poly = []
            draw_xy_seg_poly = []
            for coord in coords:
                coord = coord.text.strip('()').split(',')

                xy_seg_poly.extend([float(coord[0]), float(coord[1])])
                draw_xy_seg_poly.append([float(coord[0]), float(coord[1])])

            seg_poly = [xy_seg_poly]
            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            mask_img = np.zeros([height, width], dtype=np.uint8)
            cv2.fillPoly(mask_img, [np.array(draw_xy_seg_poly, np.int32)], 255)
            # seg_area = area(mask_img)

            mask_area = np.sum(mask_img > 0)

            abnormal_masks.append(mask_img)
            abnormal_areas.append(mask_area)

            for patch_id, pos_patch in enumerate(_sample_positive_patches(img, mask_img, mask_area, (224, 224))):
                cv2.imwrite(os.path.join('./temp/pos', f'pos_patch_{img_id}_{roi_idx}_{patch_id}.png'), pos_patch)


        for patch_id, neg_patch in enumerate(_sample_negative_patches(img, abnormal_masks, abnormal_areas, (224, 224))):
            cv2.imwrite(os.path.join('./temp/neg', f'neg_patch_{img_id}_{roi_idx}_{patch_id}.png'), neg_patch)

        if img_id > 10:
            break
