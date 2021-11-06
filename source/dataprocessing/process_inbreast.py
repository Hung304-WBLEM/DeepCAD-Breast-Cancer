import os
import glob
import pydicom
import mmcv
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

from natsort import natsorted
from scipy import stats
from pycocotools import mask as coco_api_mask


def area(_mask):
    rle = coco_api_mask.encode(np.asfortranarray(_mask))
    area = coco_api_mask.area(rle)
    return area


data_root = '/project/hnguyen/hung/Projects/Datasets/INbreast/INbreast Release 1.0'
os.makedirs(os.path.join(data_root, 'AllPNGs'), exist_ok=True)
os.makedirs(os.path.join(data_root, 'AllNormPNGs'), exist_ok=True)

images = []
annotations = []
obj_count = 0
categories = [{'id': 0, 'name': 'malignant-mass', 'supercategory': 'mass'}, {'id': 1, 'name': 'benign-mass', 'supercategory': 'mass'}]

for img_idx, dcm_path in enumerate(mmcv.track_iter_progress(natsorted(glob.glob(os.path.join(data_root, 'AllDICOMs', '*.dcm'))))):
    dcm_filename, _ = os.path.splitext(os.path.basename(dcm_path))
    mamm_img = pydicom.dcmread(dcm_path).pixel_array

    if not os.path.exists(os.path.join(data_root, 'AllPNGs', dcm_filename+'.png')):
        out = np.zeros(mamm_img.shape, np.double)
        normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
        normalized = normalized.astype(np.uint8)
        cv2.imwrite(os.path.join(data_root, 'AllPNGs', dcm_filename+'.png'), normalized)

    # For increasing contrast of mammograms
    if not os.path.exists(os.path.join(data_root, 'AllNormPNGs', dcm_filename+'.png')):
        flat_img = mamm_img.flatten()
        flat_img = flat_img[flat_img != 0]
        mode = stats.mode(flat_img).mode # ignore background value 0
        mamm_img = np.clip(mamm_img, mode-500, mode+800)

        out = np.zeros(mamm_img.shape, np.double)
        normalized = cv2.normalize(mamm_img, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) * 255
        normalized = normalized.astype(np.uint8)
        cv2.imwrite(os.path.join(data_root, 'AllNormPNGs', dcm_filename+'.png'), normalized)


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
