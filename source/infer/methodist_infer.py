from evaluation.eval_mmdet_models import get_best_ckpt
from dataprocessing.process_cbis_ddsm import read_annotation_json
from utilities.fileio import json
from config.cfg_loader import proj_paths_json
from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import glob
import numpy as np
import warnings
import math
import matplotlib
matplotlib.use('Agg')


if __name__ == '__main__':
    mmdet_root = proj_paths_json['LIB']['mmdet']
    data_root = proj_paths_json['DATA']['root']
    methodist_data = os.path.join(
        data_root, proj_paths_json['DATA']['methodist_data']['root'])

    experiment_root = proj_paths_json['EXPERIMENT']['root']
    saved_models_root = proj_paths_json['EXPERIMENT']['mmdet_processed_CBIS_DDSM']['root']
    models_root = proj_paths_json['EXPERIMENT']['mmdet_methodist']['root']

    config_file = os.path.join(
        mmdet_root, 'configs', 'cbis_ddsm_mass', 'faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py')

    # model_root = os.path.join(
        # experiment_root, saved_models_root, 'mass', 'faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm')
    model_root = os.path.join(experiment_root, models_root, 'mass', '12_08_2021', 'faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm')

    # best_ckpt_info = mmcv.load(os.path.join(model_root, 'best_ckpt.json'))

    # best_ckpt = os.path.join(model_root, 'epoch_' +
    #                          str(best_ckpt_info['epoch']) + '.pth')

    best_ckpt = os.path.join(model_root, 'best_ckpt.pth')

    Deidentified_Negative_JPEG = os.path.join(
        methodist_data, 'Deidentified_Negative_JPEG')
    Deidentified_Positive_JPEG = os.path.join(
        methodist_data, 'Deidentified_Positive_JPEG')

    print(Deidentified_Negative_JPEG)
    print(Deidentified_Positive_JPEG)

    model = init_detector(config_file, best_ckpt)

    save_root = os.path.join(experiment_root, models_root,
                             'faster_rcnn_r50_caffe_fpn_mstrain-poly_1x_ddsm')
    save_root_negative = os.path.join(save_root, 'Deidentified_Negative_JPEG')
    save_root_positive = os.path.join(save_root, 'Deidentified_Positive_JPEG')
    os.makedirs(save_root_negative, exist_ok=True)
    os.makedirs(save_root_positive, exist_ok=True)

    for patient in mmcv.track_iter_progress(glob.glob(os.path.join(Deidentified_Negative_JPEG, 'Patient*'))):
        patient_id = os.path.basename(patient)
        save_path = os.path.join(save_root_negative, patient_id)
        os.makedirs(save_path, exist_ok=True)

        for mamm in glob.glob(os.path.join(patient, 'Mammoimage', '**', '*.jpg')):
            mamm_id = os.path.basename(mamm)
            if os.path.exists(os.path.join(save_path, mamm_id)):
                continue
            result = inference_detector(model, mamm)
            model.show_result(
                mamm, result, out_file=os.path.join(save_path, mamm_id))

    for patient in mmcv.track_iter_progress(glob.glob(os.path.join(Deidentified_Positive_JPEG, 'Patient*'))):
        patient_id = os.path.basename(patient)
        save_path = os.path.join(save_root_positive, patient_id)
        os.makedirs(save_path, exist_ok=True)

        for mamm in glob.glob(os.path.join(patient, 'Mammoimage', '**', '*.jpg')):
            mamm_id = os.path.basename(mamm)
            if os.path.exists(os.path.join(save_path, mamm_id)):
                continue
            result = inference_detector(model, mamm)
            model.show_result(
                mamm, result, out_file=os.path.join(save_path, mamm_id))
