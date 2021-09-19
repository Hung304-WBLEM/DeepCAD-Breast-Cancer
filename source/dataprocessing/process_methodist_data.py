import os
import glob
import mmcv
import shutil
import warnings
import numpy as np

from utilities.fileio import json
from natsort import natsorted
from config.cfg_loader import proj_paths_json
from sklearn.model_selection import KFold
from datetime import date


def split_train_test(num_folds=2):
    methodist_data_root = os.path.join(proj_paths_json['DATA']['root'],
                                       proj_paths_json['DATA']['methodist_data']['root'])
    positive_data_root = os.path.join(methodist_data_root, proj_paths_json['DATA']['methodist_data']['Deidentified_Positive_JPEG'])
    negative_data_root = os.path.join(methodist_data_root, proj_paths_json['DATA']['methodist_data']['Deidentified_Negative_JPEG'])

    positive_mamms = [[json_file for json_file in natsorted(glob.glob(os.path.join(patient, 'Mammoimage', '*', '*.json')))]
                      for patient in natsorted(glob.glob(os.path.join(positive_data_root, 'Patient_*')))]
    negative_mamms = [[json_file for json_file in natsorted(glob.glob(os.path.join(patient, 'Mammoimage', '*', '*.json')))]
                      for patient in natsorted(glob.glob(os.path.join(negative_data_root, 'Patient_*')))]

    flat_positive_mamms = [patient_mamm for patient in positive_mamms for patient_mamm in patient]
    flat_negative_mamms = [patient_mamm for patient in negative_mamms for patient_mamm in patient]

    all_mamms = np.array(flat_positive_mamms + flat_negative_mamms)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(all_mamms):
        yield all_mamms[train_idx], all_mamms[test_idx]

    # positive_len = len(flat_positive_mamms)
    # negative_len = len(flat_negative_mamms)

    # train_ratio, test_ratio = 0.9, 0.1

    # train_mamms = flat_positive_mamms[:int(positive_len*train_ratio)] + flat_negative_mamms[:int(negative_len*train_ratio)]
    # test_mamms = flat_positive_mamms[int(positive_len*train_ratio):] + flat_negative_mamms[int(negative_len*train_ratio):]

    # return train_mamms, test_mamms


def convert_methodist_to_coco(json_split, categories, data_root, out_file):
    save_path = os.path.join(data_root, out_file)
    if os.path.exists(save_path):
        warnings.warn(f"{save_path} has already existed")
        return

    images = []
    annotations = []
    obj_count = 0

    # save_root = '/project/hnguyen/hung/Projects/Datasets/methodist_data'
    # train_save_root = os.path.join(save_root, 'train')
    # test_save_root = os.path.join(save_root, 'test')

    # train_split, test_split = split_train_test()
    # if os.path.basename(data_root) == 'train':
    #     split = train_split
    # elif os.path.basename(data_root) == 'test':
    #     split = test_split

    for idx, mamm in enumerate(json_split):
        file_path = '/'.join(mamm.split('/')[-5:])
        dir_path = os.path.split(file_path)[0]

        # Create images
        json_data = json.read(mamm)
        image_filename = os.path.join(dir_path, json_data['imagePath'])
        image_height = json_data['imageHeight']
        image_width = json_data['imageWidth']

        source_image_path = os.path.join(os.path.split(mamm)[0], json_data['imagePath'])
        # target_image_path = os.path.join(train_save_root, image_filename)
        target_image_path = os.path.join(data_root, image_filename)

        os.makedirs(os.path.split(target_image_path)[0], exist_ok=True)

        shutil.copyfile(source_image_path, target_image_path)

        images.append(dict(
            id=idx,
            file_name = image_filename,
            height = image_height,
            width = image_width
        ))

        # Create annotations
        label = image_filename.split('/')[0]
        if label == 'Deidentified_Positive_JPEG':
            cat_id = 0
        elif label == 'Deidentified_Negative_JPEG':
            cat_id = 1
        else:
            raise ValueError("label is not defined")

        for bbox in json_data['shapes']:
            coords = bbox['points']
            x_min, y_min = coords[0][0], coords[0][1]
            x_max, y_max = coords[1][0], coords[1][1]

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=cat_id,
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
                area = (x_max-x_min)*(y_max-y_min),
                segmentation=[],
                iscrowd=0
            )

            annotations.append(data_anno)
            obj_count+=1


    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    mmcv.dump(coco_format_json, os.path.join(data_root, out_file))


if __name__ == '__main__':
    categories = [{'id': 0, 'name': 'malignant-mass', 'supercategory': 'mass'},
                  {'id': 1, 'name': 'benign-mass', 'supercategory': 'mass'}]

    result_root = os.path.join('/home/hqvo2/Projects/Breast_Cancer/data/methodist_data/train_test_folds', date.today().strftime("%d_%m_%Y"))

    train_folds_root = os.path.join(result_root, 'train_folds')
    test_folds_root = os.path.join(result_root, 'test_folds')

    for idx, (train_split, test_split) in enumerate(split_train_test(num_folds=10)):
        print(train_split, test_split)
        convert_methodist_to_coco(train_split, categories, os.path.join(train_folds_root, f'train_fold_{idx}'), 'train.json')
        convert_methodist_to_coco(test_split, categories, os.path.join(test_folds_root, f'test_fold_{idx}'), 'test.json')

    '''
    data_root = '/project/hnguyen/hung/Projects/Datasets/methodist_data/train'
    convert_methodist_to_coco(categories, 'train.json', data_root)

    data_root = '/project/hnguyen/hung/Projects/Datasets/methodist_data/test'
    convert_methodist_to_coco(categories, 'test.json', data_root)
    '''
