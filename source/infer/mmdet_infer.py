import mmcv
import os
import glob
import numpy as np
import warnings
import math

from mmdet.apis import init_detector, inference_detector
from config.cfg_loader import proj_paths_json
from utils.fileio import json
from dataprocessing.process_cbis_ddsm import read_annotation_json
from evaluation.eval_mmdet_models import get_best_ckpt


def get_predictions(config_file, checkpoint_file, data_root, data_json, save_root):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print(model)

    img_annotations, categories = read_annotation_json(
        os.path.join(data_root, data_json))

    for img, anns in mmcv.track_iter_progress(img_annotations):
        img_path = os.path.join(data_root, img['file_name'])
        file_name, _ = os.path.splitext(os.path.basename(img['file_name']))
        save_path = os.path.join(save_root, f"{file_name}.txt")

        if os.path.exists(save_path):
            warnings.warn(f"{save_path} has already existed! Skip!")
            continue

        result = inference_detector(model, img_path)
        print('result', result)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)  # format: x1, y1, x2, y2, score

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        with open(save_path, 'w') as f:
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2, s = (str(el) for el in bbox)
                c = [el['name'] for el in categories if el['id'] == label][0]
                f.write(' '.join((c, s, x1, y1, x2, y2, '\n')))

        break


if __name__ == '__main__':
    mmdet_root = proj_paths_json['LIB']['mmdet']
    experiment_root = proj_paths_json['EXPERIMENT']['root']
    saved_models_root = proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['root']

    for config in glob.glob(os.path.join(mmdet_root, 'configs', 'ddsm', '*.py')):
        config_name, _ = os.path.splitext(os.path.basename(config))

        if config_name != 'faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm':
            continue
        print(config_name)

        config_file = os.path.join(
            mmdet_root, 'configs', 'ddsm', f'{config_name}.py')
        model_root = os.path.join(
            experiment_root, saved_models_root, f'{config_name}')
        best_ckpt_info = get_best_ckpt(model_root, metric='bbox_mAP')
        mmcv.dump(best_ckpt_info, os.path.join(model_root, 'best_ckpt.json'))

        best_ckpt = os.path.join(model_root, 'epoch_' +
                                 str(best_ckpt_info['epoch']) + '.pth')

        data_root = proj_paths_json['DATA']['root']
        processed_cbis_ddsm_root = os.path.join(
            data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

        mass_train_root = os.path.join(
            processed_cbis_ddsm_root, 'mass', 'train')
        mass_test_root = os.path.join(processed_cbis_ddsm_root, 'mass', 'test')
        save_root = os.path.join(experiment_root,
                                 saved_models_root,
                                 f'{config_name}', 'detections')
        save_root_train = os.path.join(save_root, 'train')
        save_root_test = os.path.join(save_root, 'test')

        if not os.path.exists(save_root_train):
            os.makedirs(save_root_train, exist_ok=True)
        if not os.path.exists(save_root_test):
            os.makedirs(save_root_test, exist_ok=True)
        get_predictions(config_file, checkpoint_file=best_ckpt,
                        data_root=mass_train_root, data_json='annotation_coco_with_classes.json',
                        save_root=save_root_train)

        get_predictions(config_file, checkpoint_file=best_ckpt,
                        data_root=mass_test_root, data_json='annotation_coco_with_classes.json',
                        save_root=save_root_test)

        break
