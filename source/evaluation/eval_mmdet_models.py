import os
import glob
import json


def get_best_ckpt(model_dir, metric):
    '''
    Parameters:
    model_dir - path to the directory of saved ckpts
    metric - metric used for selecting best ckpt (e.g.: 'bbox_mAP', 'segm_mAP')
    '''
    print(f'Get best checkpoint from {model_dir}')
    eps = []
    for log_file in glob.glob(os.path.join(model_dir, '*.log.json')):
        for line in open(log_file, 'r'):
            json_data = json.loads(line)

            if 'mode' in json_data and json_data['mode'] == 'val':
                eps.append(json_data)

    if metric in ['bbox_mAP', 'segm_mAP']:  # larger value is better
        eps = sorted(eps, key=lambda x: (x[metric], -x['epoch']))
    else:  # smaller value is better
        eps = sorted(eps, key=lambda x: (-x[metric], -x['epoch']))

    return eps[-1]


if __name__ == '__main__':
    best_ckpt_info = get_best_ckpt('../../experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm',
                                   metric='bbox_mAP')
    print(best_ckpt_info)
