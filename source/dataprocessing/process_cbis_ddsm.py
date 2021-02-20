import os
import glob
import numpy as np
import cv2
import mmcv
import pandas
import warnings
import math

from config.cfg_loader import proj_paths_json
from pycocotools import mask as coco_api_mask
from utils.detectutil import bbox_util


def convert_npz_to_png(data_path):
    for dir in glob.glob(os.path.join(data_path, '*')):
        num_files = len(glob.glob(os.path.join(dir, '*')))
        if num_files < 2:
            print(os.path.basename(dir), num_files)
            continue

        save_path = os.path.join(dir, os.path.basename(dir)+'.png')
        if not os.path.exists(save_path):
            mamm_img = np.load(os.path.join(dir, "image.npz"),
                               allow_pickle=True)["image"]
            cv2.imwrite(save_path, mamm_img)


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


def convert_ddsm_to_coco(categories, out_file, data_root, annotation_filename, extend_bb_ratio=None, keep_org_boxes=False):
    save_path = os.path.join(data_root, out_file)
    if os.path.exists(save_path):
        warnings.warn(f"{save_path} has already existed")
        return

    images = []
    annotations = []
    obj_count = 0

    df = pandas.read_csv(os.path.join(data_root, annotation_filename))

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        img_path = os.path.join(dir_path, filename + '.png')
        if not os.path.exists(img_path):
            continue
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=os.path.join(filename, filename+'.png'),
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []

        for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
            while True:
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

            mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
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


def get_patches_data(mass_shape_root, mass_margins_root, data_root, annotation_filename):
    '''
    '''
    df = pandas.read_csv(os.path.join(data_root, annotation_filename))

    for idx, dir_path in enumerate(mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '*')))):
        filename = os.path.basename(dir_path)
        img_path = os.path.join(dir_path, filename + '.png')
        if not os.path.exists(img_path):
            continue

        img = mmcv.imread(img_path)

        for roi_idx, mask_path in enumerate(glob.glob(os.path.join(dir_path, 'mask*.npz'))):
            while True:
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

            mask_arr = np.load(mask_path, allow_pickle=True)["mask"]
            seg_poly = mask2polygon(mask_arr)
            seg_area = area(mask_arr)

            flat_seg_poly = [el for sublist in seg_poly for el in sublist]
            px = flat_seg_poly[::2]
            py = flat_seg_poly[1::2]
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            lesion_patch = img[y_min:(y_max+1), x_min:(x_max+1)]

            mass_shape = rslt_df['mass shape'].to_numpy()[0]
            if isinstance(mass_shape, str):
                mass_shape_save_path = os.path.join(
                    mass_shape_root, mass_shape)
                save_mass_shape_lesion_path = os.path.join(
                    mass_shape_save_path,  f'{filename}_{roi_idx}.png')
                if not os.path.exists(save_mass_shape_lesion_path):
                    os.makedirs(mass_shape_save_path, exist_ok=True)
                    cv2.imwrite(save_mass_shape_lesion_path, lesion_patch)

            mass_margins = rslt_df['mass margins'].to_numpy()[0]
            if isinstance(mass_margins, str):
                mass_margins_save_path = os.path.join(
                    mass_margins_root, mass_margins)

                save_mass_margins_lesion_path = os.path.join(
                    mass_margins_save_path,  f'{filename}_{roi_idx}.png')
                if not os.path.exists(save_mass_margins_lesion_path):
                    os.makedirs(mass_margins_save_path, exist_ok=True)
                    cv2.imwrite(save_mass_margins_lesion_path, lesion_patch)


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

    convert_npz_to_png(data_path=mass_train_root)
    convert_npz_to_png(data_path=mass_test_root)

    categories = [{'id': 0, 'name': 'malignant-mass', 'supercategory': 'mass'}, {'id': 1, 'name': 'benign-mass', 'supercategory': 'mass'}]

    # Default bbox size
    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes.json',
                         data_root=mass_train_root,
                         annotation_filename='mass_case_description_train_set.csv')

    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes.json',
                         data_root=mass_test_root,
                         annotation_filename='mass_case_description_test_set.csv')

    # Extend bbox size by 0.3
    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.3.json',
                         data_root=mass_train_root,
                         annotation_filename='mass_case_description_train_set.csv',
                         extend_bb_ratio=0.3)

    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.3.json',
                         data_root=mass_test_root,
                         annotation_filename='mass_case_description_test_set.csv',
                         extend_bb_ratio=0.3)

    # Extend bbox size by 0.2
    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.2.json',
                         data_root=mass_train_root,
                         annotation_filename='mass_case_description_train_set.csv',
                         extend_bb_ratio=0.2)

    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.2.json',
                         data_root=mass_test_root,
                         annotation_filename='mass_case_description_test_set.csv',
                         extend_bb_ratio=0.2)

    # Extend bbox size by 0.1
    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes_extend_bbox_0.1.json',
    #                      data_root=mass_train_root,
    #                      annotation_filename='mass_case_description_train_set.csv',
    #                      extend_bb_ratio=0.1)

    # convert_ddsm_to_coco(categories=categories,
    #                      out_file='annotation_coco_with_classes_extend_bbox_0.1.json',
    #                      data_root=mass_test_root,
    #                      annotation_filename='mass_case_description_test_set.csv',
    #                      extend_bb_ratio=0.1)

    # Extend bbox size by 0.2 and keep original bbox
    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.2_aug.json',
                         data_root=mass_train_root,
                         annotation_filename='mass_case_description_train_set.csv',
                         extend_bb_ratio=0.2, keep_org_boxes=True)

    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes_extend_bbox_0.2_aug.json',
                         data_root=mass_test_root,
                         annotation_filename='mass_case_description_test_set.csv',
                         extend_bb_ratio=0.2, keep_org_boxes=True)

    ############## Extract Lesion Patches ##############
    # mass_shape_root = os.path.join(
    #     processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_shape_root'])
    # mass_margins_root = os.path.join(
    #     processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_margins_root'])

    # get_patches_data(os.path.join(mass_shape_root, 'train'), os.path.join(mass_margins_root, 'train'),
    #                  data_root=mass_train_root, annotation_filename='mass_case_description_train_set.csv')
    # get_patches_data(os.path.join(mass_shape_root, 'val'), os.path.join(mass_margins_root, 'val'),
    #                  data_root=mass_test_root, annotation_filename='mass_case_description_test_set.csv')

    ################################################################
    ##################### CALCIFICATION LESIONS ####################
    ################################################################

    ################# Process Calcification Lesions Mammogram ################
    calc_train_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'train')
    calc_test_root = os.path.join(processed_cbis_ddsm_root, 'calc', 'test')

    convert_npz_to_png(data_path=calc_train_root)
    convert_npz_to_png(data_path=calc_test_root)

    categories = [{'id': 0, 'name': 'malignant-calc', 'supercategory': 'calcification'}, {'id': 1, 'name': 'benign-calc', 'supercategory': 'calcification'}]

    # Default bbox size
    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes.json',
                         data_root=calc_train_root,
                         annotation_filename='calc_case_description_train_set.csv')

    convert_ddsm_to_coco(categories=categories,
                         out_file='annotation_coco_with_classes.json',
                         data_root=calc_test_root,
                         annotation_filename='calc_case_description_test_set.csv')

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
