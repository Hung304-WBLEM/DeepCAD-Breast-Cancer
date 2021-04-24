import os
import glob
import json

from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter

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

    for ep in eps:
        print(ep)
    if metric in ['bbox_mAP', 'bbox_mAP_50', 'segm_mAP']:  # larger value is better
        eps = sorted(eps, key=lambda x: (x[metric], -x['epoch']))
    else:  # smaller value is better
        eps = sorted(eps, key=lambda x: (-x[metric], -x['epoch']))

    return eps[-1]


def train_val_log(model_dir):
    writer = SummaryWriter(os.path.join(model_dir, 'tensorboard_logs'))

    for log_file in glob.glob(os.path.join(model_dir, '*.log.json')):
        for line in open(log_file, 'r'):
            json_data = json.loads(line)

            if 'mode' in json_data:
                epoch = json_data['epoch']

                if json_data['mode'] == 'train':
                    train_loss = json_data['loss']

                    writer.add_scalar('train loss', train_loss, epoch)

                elif json_data['mode'] == 'val':
                    val_bbox_mAP_50 = json_data['bbox_mAP_50']

                    writer.add_scalar('val bbox_mAP_50', val_bbox_mAP_50, epoch)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--parent_root",
                      help="root to the parent directory which contains all the configurations")
    parser.add_option("--metric",
                      help="metric to select best ckpt. Could be: bbox_mAP, segm_mAP, bbox_mAP_50")
    options, _ = parser.parse_args()

    # Mass Configs
    for config_dir in glob.glob(os.path.join(options.parent_root, '*')):
        #######################
        # Get best checkpoint #
        #######################
        best_ckpt_info = get_best_ckpt(model_dir=config_dir,
                                       metric=options.metric)
        best_epoch = best_ckpt_info['epoch']
        current_working_dir = os.getcwd()
        os.chdir(config_dir)

        if not os.path.exists('best_ckpt.pth'):
            os.symlink(f'epoch_{best_epoch}.pth', 'best_ckpt.pth')
        else:
            print(f'Best checkpoint has already existed in {config_dir}')

        os.chdir(current_working_dir)

        ######################
        # plot train val log #
        ######################
        train_val_log(model_dir=config_dir)

    # train_val_log(model_dir="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/calc/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm")
