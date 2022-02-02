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


def area(_mask):
    rle = coco_api_mask.encode(np.asfortranarray(_mask))
    area = coco_api_mask.area(rle)
    return area


def increase_constrast_by_clip(mamm_img, lowerbound_clip_rate, upperbound_clip_rate):
    flat_img = mamm_img.flatten()
    flat_img = flat_img[flat_img != 0]
    mode = stats.mode(flat_img).mode # ignore background value 0
    
    hist, _ = np.histogram(mamm_img.ravel(),256,[0,256])
    
    lowerbound_total_px = 0
    lowerbound_mode_total_px = np.sum(hist[1:mode[0]]) # ignore background px
    selected_lowerbound_color = None
    for i in range(1, mode[0]): # ignore background px
        lowerbound_total_px = np.sum(hist[1:(i+1)])      
        if lowerbound_total_px / lowerbound_mode_total_px >= lowerbound_clip_rate:
            selected_lowerbound_color = i
            break
        
    upperbound_total_px = 0
    upperbound_mode_total_px = np.sum(hist[(mode[0]+1):])
    selected_upperbound_color = None
    for i in range(255, mode[0]+1, -1):
        upperbound_total_px = np.sum(hist[i:256])
        if upperbound_total_px / upperbound_mode_total_px >= upperbound_clip_rate:
            selected_upperbound_color = i
            break
            
    # if pmin > mode:
    #   print('pmin is larger than mode')
    #   pmin = mode
    # if pmax > 255 - mode:
    #   print('pmax is larger than 255-mode')
    #   pmax = 255 - mode

    mamm_img = np.clip(mamm_img, selected_lowerbound_color, selected_upperbound_color)

    out = np.zeros(mamm_img.shape, np.double)
    normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
    normalized = normalized.astype(np.uint8)

    return normalized


def convert_dicom_to_png(data_root):
    os.makedirs(os.path.join(data_root, 'AllPNGs'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'AllNormPNGs'), exist_ok=True)

    for img_idx, dcm_path in enumerate(mmcv.track_iter_progress(natsorted(glob.glob(os.path.join(data_root, 'AllDICOMs', '*.dcm'))))):
        dcm_filename, _ = os.path.splitext(os.path.basename(dcm_path))
        mamm_img = pydicom.dcmread(dcm_path).pixel_array

        if not os.path.exists(os.path.join(data_root, 'AllPNGs', dcm_filename+'.png')):
            out = np.zeros(mamm_img.shape, np.double)
            normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
            normalized = normalized.astype(np.uint8)
            cv2.imwrite(os.path.join(data_root, 'AllPNGs', dcm_filename+'.png'), normalized)

        # For increasing contrast of mammograms (Follow the FRCNN paper)
        if not os.path.exists(os.path.join(data_root, 'AllNormPNGs', dcm_filename+'.png')):
            flat_img = mamm_img.flatten()
            flat_img = flat_img[flat_img != 0]
            mode = stats.mode(flat_img).mode # ignore background value 0
            mamm_img = np.clip(mamm_img, mode-500, mode+800)

            out = np.zeros(mamm_img.shape, np.double)
            normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
            normalized = normalized.astype(np.uint8)
            cv2.imwrite(os.path.join(data_root, 'AllNormPNGs', dcm_filename+'.png'), normalized)


def get_inbreast_lesion_pathology(data_root, mass_save_root,
                                  calc_save_root, cluster_save_root,
                                  distortion_save_root,
                                  spiculated_save_root,
                                  asymetry_save_root,
                                  background_save_root=None,
                                  patch_ext='center'):

    for img_idx, img_path in enumerate(mmcv.track_iter_progress(natsorted(glob.glob(os.path.join(data_root, 'AllNormPNGs', '*.png'))))):
        filename, _ = os.path.splitext(os.path.basename(img_path))

        xml_path = os.path.join(data_root, 'AllXML', filename.split('_')[0] + '.xml') 

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        _, _, _, num_rois, _, rois = root.getchildren()[0].getchildren()[1].getchildren()[0].getchildren()

        img = mmcv.imread(img_path)
        resized_img = cv2.resize(img, (896, 1152))
        height, width = img.shape[:2]
        num_rois = num_rois.text

        # images.append(dict(
        #     id=img_idx,
        #     file_name=img_name,
        #     height=height,
        #     width=width
        # ))

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
                if float(_area.text) == 0:
                    continue
                save_root = calc_save_root
            elif name.text in ['Mass']:
                save_root = mass_save_root
            elif name.text in ['Cluster']:
                save_root = cluster_save_root
            elif name.text in ['Distortion']:
                save_root = distortion_save_root
            elif name.text in ['Spiculated region',
                               'Espiculated Region', 'Spiculated Region']:
                save_root = spiculated_save_root
            elif name.text in ['Asymmetry', 'Assymetry']:
                save_root = asymetry_save_root
            else:
                continue

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

                pad_width_size = max(patch_size - (new_x_max - new_x_min), 0)
                pad_height_size = max(patch_size - (new_y_max - new_y_min), 0)

                lesion_patch = np.pad(lesion_patch,
                                    [(int(pad_height_size//2),
                                      int(pad_height_size - int(pad_height_size//2))),
                                     (int(pad_width_size//2),
                                      int(pad_width_size - int(pad_width_size//2))),
                                     (0, 0)], 'constant')

            if cat_id == 0:
                save_path = os.path.join(save_root, 'MALIGNANT')
            elif cat_id == 1:
                save_path = os.path.join(save_root, 'BENIGN')

            os.makedirs(save_path, exist_ok=True)

            if patch_ext in ['exact', 'center']:
                cv2.imwrite(os.path.join(save_path, f'{filename}_{roi_idx}.png'), lesion_patch)
            elif patch_ext == 'random':
                for patch_id, pos_patch in enumerate(_sample_positive_patches(resized_img,
                                                                              resized_mask_img,
                                                                              mask_area,
                                                                              (224, 224))):
                    cv2.imwrite(os.path.join(save_path,
                                             f'{filename}_{roi_idx}_patch_{patch_id}.png'),
                                pos_patch)

        if patch_ext == 'random':
            os.makedirs(background_save_root, exist_ok=True)
            for patch_id, neg_patch in enumerate(_sample_negative_patches(resized_img,
                                                                          abnormal_masks,
                                                                          abnormal_areas,
                                                                          (224, 224))):
                cv2.imwrite(os.path.join(background_save_root,
                                         f'{filename}_background_{patch_id}.png'),
                            neg_patch)

    

def convert_inbreast_to_coco(data_root, categories):
    images = []
    annotations = []
    obj_count = 0

    for img_idx, dcm_path in enumerate(mmcv.track_iter_progress(natsorted(glob.glob(os.path.join(data_root, 'AllDICOMs', '*.dcm'))))):
        dcm_filename, _ = os.path.splitext(os.path.basename(dcm_path))

        xml_path = os.path.join(data_root, 'AllXML', dcm_filename.split('_')[0] + '.xml') 

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        _, _, _, num_rois, _, rois = root.getchildren()[0].getchildren()[1].getchildren()[0].getchildren()

        img_name = dcm_filename + '.png'
        mamm_img = mmcv.imread(os.path.join(data_root, 'AllPNGs', img_name))
        height, width = mamm_img.shape[:2]
        num_rois = num_rois.text

        images.append(dict(
            id=img_idx,
            file_name=img_name,
            height=height,
            width=width
        ))

        labels_data = pd.read_csv(os.path.join(data_root, 'INbreast.csv'), sep=';')
        label = labels_data[labels_data['File Name'] == int(dcm_filename.split('_')[0])]['Bi-Rads'].iloc[0]
        if label == '3':
            print('Label is probably benign')
            continue
        elif label in ['1', '2']:
            label = 'benign-mass'
            cat_id = 1
        elif label in ['4a', '4b', '4c', '5', '6']:
            label = 'malignant-mass'
            cat_id = 0
        else:
            print(label)
            raise ValueError

        for roi in rois.getchildren():
            _, _area, _, center, _, dev, _, idx_in_img, _, _max, _, mean, _, _min, _, name, _, num_pts, _, pointmm, _, pointpx, _, total, _, _type = roi.getchildren()

            if name.text not in ['Mass', 'Spiculated Region']:
                continue

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
            seg_area = area(mask_img)

            # cv2.polylines(mamm_img, [np.array(draw_xy_seg_poly, np.int32)], isClosed=True, color=(0, 255, 0), thickness=4)
            # cv2.imwrite('./mask_img.png', mask_img)
            # cv2.imwrite('./mamm_img.png', mamm_img)


            data_anno = dict(
                image_id = img_idx,
                id=obj_count,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=seg_area,
                segmentation=seg_poly,
                iscrowd=0
            )

            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    mmcv.dump(coco_format_json, os.path.join(data_root, 'inbreast_annotation_coco.json'))


if __name__ == '__main__':
    inbreast = proj_paths_json['DATA']['INbreast']
    data_root = os.path.join(proj_paths_json['DATA']['root'], inbreast['root'])

    categories = [{'id': 0, 'name': 'malignant-mass', 'supercategory': 'mass'},
                  {'id': 1, 'name': 'benign-mass', 'supercategory': 'mass'}]

    # convert_dicom_to_png(data_root)

    # convert_inbreast_to_coco(data_root, categories)

    # Extract Lesion (Center)
    mass_save_root = os.path.join(data_root, inbreast['mass_feats']['mass_pathology'])
    calc_save_root = os.path.join(data_root, inbreast['calc_feats']['calc_pathology'])
    cluster_save_root = os.path.join(data_root, inbreast['cluster_feats']['cluster_pathology'])
    distortion_save_root = os.path.join(data_root, inbreast['distortion_feats']['distortion_pathology'])
    spiculated_save_root = os.path.join(data_root, inbreast['spiculated_feats']['spiculated_pathology'])
    asymetry_save_root = os.path.join(data_root, inbreast['asymetry_feats']['asymetry_pathology'])

    get_inbreast_lesion_pathology(data_root, mass_save_root,
                                  calc_save_root, cluster_save_root,
                                  distortion_save_root,
                                  spiculated_save_root,
                                  asymetry_save_root,
                                  )

    # Extract Lesion (Random)
    mass_save_root = os.path.join(data_root, inbreast['mass_feats']['aug_mass_pathology'])
    calc_save_root = os.path.join(data_root, inbreast['calc_feats']['aug_calc_pathology'])
    cluster_save_root = os.path.join(data_root, inbreast['cluster_feats']['aug_cluster_pathology'])
    distortion_save_root = os.path.join(data_root, inbreast['distortion_feats']['aug_distortion_pathology'])
    spiculated_save_root = os.path.join(data_root, inbreast['spiculated_feats']['aug_spiculated_pathology'])
    asymetry_save_root = os.path.join(data_root, inbreast['asymetry_feats']['aug_asymetry_pathology'])
    background_save_root = os.path.join(data_root, inbreast['background']['bg_tfds'])

    get_inbreast_lesion_pathology(data_root, mass_save_root,
                                  calc_save_root, cluster_save_root,
                                  distortion_save_root,
                                  spiculated_save_root,
                                  asymetry_save_root,
                                  background_save_root,
                                  patch_ext='random'
                                  )
    remove_background_images(background_save_root)
