import os
import glob

from utilities.fileio import json
from natsort import natsorted


def count_num_images_and_annotations_gt_json(gt_json_path):
    json_content = json.read(gt_json_path)
    return len(json_content['images']), len(json_content['annotations'])


if __name__ == '__main__':
    test_gt_path = "/home/hqvo2/Projects/Breast_Cancer/data/methodist_data/train_test_folds/12_08_2021/test_folds/"

    all_test_folds = natsorted(glob.glob(os.path.join(test_gt_path, 'test_fold_*')))
    total_images = 0
    total_annotations = 0

    images = json.read(os.path.join(all_test_folds[0], 'test.json'))['images']
    annotations = json.read(os.path.join(all_test_folds[0], 'test.json'))['annotations']
    categories = json.read(os.path.join(all_test_folds[0], 'test.json'))['categories']

    test_pred_path = "/home/hqvo2/Projects/Breast_Cancer/experiments/methodist_data_detection/mass/12_08_2021/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm"
    all_pred_test_folds = natsorted(glob.glob(os.path.join(test_pred_path, 'fold_*')))
    predictions = json.read(os.path.join(all_pred_test_folds[0], 'test_bboxes.bbox.json'))

    for idx in range(1, len(all_test_folds)):
        prev_num_images, prev_num_anno = count_num_images_and_annotations_gt_json(os.path.join(all_test_folds[idx-1], 'test.json'))
        total_images += prev_num_images
        total_annotations += prev_num_anno

        ########
        cur_json = json.read(os.path.join(all_test_folds[idx], 'test.json'))

        for img_idx in range(len(cur_json['images'])):
            cur_json['images'][img_idx]['id'] += total_images

        for ann_idx in range(len(cur_json['annotations'])):
            cur_json['annotations'][ann_idx]['image_id'] += total_images
            cur_json['annotations'][ann_idx]['id'] += total_annotations

        images.extend(cur_json['images'])
        annotations.extend(cur_json['annotations'])
        ###########

        cur_pred_json = json.read(os.path.join(all_pred_test_folds[idx], 'test_bboxes.bbox.json'))

        for pred_idx in range(len(cur_pred_json)):
            cur_pred_json[pred_idx]['image_id'] += total_images

        predictions.extend(cur_pred_json)

    gt_json_content = {'images': images, 'annotations': annotations, 'categories': categories}
    json.write(gt_json_content, os.path.join(test_gt_path, 'concat_gt.json'))

    json.write(predictions, os.path.join(test_pred_path, 'concat_pred.json'))




    # test_pred_path = "/home/hqvo2/Projects/Breast_Cancer/experiments/methodist_data_detection/mass/12_08_2021/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/fold_0/test_bboxes.bbox.json"
    # test_pred_json = json.read(test_pred_path)
    # print(test_pred_json)
