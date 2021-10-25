import os
import subprocess
import sys
import glob
import importlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')

from torch.utils.tensorboard import SummaryWriter
from natsort import natsorted


def visualize_img_tensorboard(save_path, log_title, num_imgs=None):
    for idx, img_path in enumerate(natsorted(glob.glob(os.path.join(save_path, '*.png')))):
        if num_imgs is not None and idx == num_imgs:
            break

        fig = plt.figure(figsize=(10, 18))
        img = mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        plt.title(os.path.basename(img_path))
        plt.tight_layout()
        plt.axis('off')
        writer.add_figure(log_title,
                        fig, global_step=idx)
        plt.close()

def visualize_curve_tensorboard(img_path):
    fig = plt.figure(figsize=(7, 7))
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()
    plt.tight_layout()
    plt.axis('off')
    writer.add_figure(os.path.basename(img_path),
                      fig, global_step=0)
    plt.close()
    


if __name__ == '__main__':
    mmdet_root = '/home/hqvo2/Projects/Breast_Cancer/libs/mmdetection'
    eval_src_root = '/home/hqvo2/Projects/Breast_Cancer/source/evaluation'

    save_root_list = [
        # '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/deformable_detr_r50_16x2_50e_ddsm',
        # '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/deformable_detr_refine_r50_16x2_50e_ddsm',
        # '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/deformable_detr_twostage_refine_r50_16x2_50e_ddsm',
        '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_ddsm',
        # '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm_albu',
        # '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm',
        '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass/faster_rcnn_x101_64x4d_fpn_1x_ddsm'
    ]

    for save_root in save_root_list:
        writer = SummaryWriter(os.path.join(save_root, 'tf_logs'))

        os.chdir(mmdet_root)

        config_file = glob.glob(os.path.join(save_root, '*.py'))[0] # config file is the only python file in save root
        best_ckpt = glob.glob(os.path.join(save_root, 'best*.pth'))[0]


        # Get predicted bboxes as a pickle file
        if not os.path.exists(os.path.join(save_root, 'result.pkl')):
            process = subprocess.Popen(['sh', 'tools/dist_test.sh',
                                        config_file,
                                        best_ckpt,
                                        '4', '--out',
                                        os.path.join(save_root, 'result.pkl')])
            process.wait()

        
        # Get predicted bboxes as a json file
        if not os.path.exists(os.path.join(save_root, 'result.bbox.json')):
            process = subprocess.Popen(['sh',  'tools/dist_test.sh',
                                        config_file,
                                        best_ckpt,
                                        '4', '--format-only', '--eval-options',
                                        'jsonfile_prefix=' + os.path.join(save_root, 'result')])
            process.wait()

            os.chdir(save_root)
            sys.path.append(save_root)
            # from faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm import DDSM_TEST_ANNOTATION
            config_mod = importlib.import_module(os.path.basename(config_file.split('.')[0]))

            os.chdir(eval_src_root)
            process = subprocess.Popen(['python', 'plot_eval_curve.py',
                                        '-gt', config_mod.TEST_ANNOTATION,
                                        '-p', os.path.join(save_root, 'result.bbox.json'),
                                        '--log_title', 'test',
                                        '-bb', 'all',
                                        '-s', os.path.join(save_root, 'curves')])
            process.wait()


        os.chdir(mmdet_root)

        # To visualize images with the highest and lowest detection scores. This is to debug your model.
        if not os.path.exists(os.path.join(save_root, 'results')):
            process = subprocess.Popen(['python', 'tools/analysis_tools/analyze_results.py',
                                        config_file,
                                        os.path.join(save_root, 'result.pkl'),
                                        os.path.join(save_root, 'results')])
            process.wait()

            # for idx, img_path in enumerate(glob.glob(os.path.join(save_root, 'results', 'good', '*.png'))):
            #     fig = plt.figure(figsize=(10, 18))
            #     img = mpimg.imread(img_path)
            #     imgplot = plt.imshow(img)
            #     plt.show()
            #     plt.title(os.path.basename(img_path))
            #     plt.tight_layout()
            #     plt.axis('off')
            #     writer.add_figure('Debugging Visualization - Good Cases',
            #                     fig, global_step=idx)
            #     plt.close()

            # for idx, img_path in enumerate(glob.glob(os.path.join(save_root, 'results', 'bad', '*.png'))):
            #     fig = plt.figure(figsize=(10, 18))
            #     img = mpimg.imread(img_path)
            #     imgplot = plt.imshow(img)
            #     plt.show()
            #     plt.title(os.path.basename(img_path))
            #     plt.tight_layout()
            #     plt.axis('off')
            #     writer.add_figure('Debugging Visualization - Bad Cases',
            #                     fig, global_step=idx)
            #     plt.close()
            visualize_img_tensorboard(save_path=os.path.join(save_root, 'results', 'good'),
                                      log_title='Debugging Visualization - Good Cases')
            visualize_img_tensorboard(save_path=os.path.join(save_root, 'results', 'bad'),
                                      log_title='Debugging Visualization - Bad Cases')

        # To visualize images with the highest and lowest detection scores. This is to debug your model (With threshold)
        if not os.path.exists(os.path.join(save_root, 'results_threshold')):
            process = subprocess.Popen(['python', 'tools/analysis_tools/analyze_results.py',
                                        config_file,
                                        os.path.join(save_root, 'result.pkl'),
                                        os.path.join(save_root, 'results_threshold'),
                                        '--show-score-thr', '0.3'])
            process.wait()

            # for idx, img_path in enumerate(glob.glob(os.path.join(save_root, 'results_threshold', 'good', '*.png'))):
            #     fig = plt.figure(figsize=(10, 18))
            #     img = mpimg.imread(img_path)
            #     imgplot = plt.imshow(img)
            #     plt.show()
            #     plt.title(os.path.basename(img_path))
            #     plt.tight_layout()
            #     plt.axis('off')
            #     writer.add_figure('Debugging Visualization - Good Cases (thres 0.3)',
            #                     fig, global_step=idx)
            #     plt.close()

            # for idx, img_path in enumerate(glob.glob(os.path.join(save_root, 'results_threshold', 'bad', '*.png'))):
            #     fig = plt.figure(figsize=(10, 18))
            #     img = mpimg.imread(img_path)
            #     imgplot = plt.imshow(img)
            #     plt.show()
            #     plt.title(os.path.basename(img_path))
            #     plt.tight_layout()
            #     plt.axis('off')
            #     writer.add_figure('Debugging Visualization - Bad Cases (thres 0.3)',
            #                     fig, global_step=idx)
            #     plt.close()

            visualize_img_tensorboard(save_path=os.path.join(save_root, 'results_threshold', 'good'),
                                      log_title='Debugging Visualization - Good Cases (thres 0.3)')
            visualize_img_tensorboard(save_path=os.path.join(save_root, 'results_threshold', 'bad'),
                                      log_title='Debugging Visualization - Bad Cases (thres 0.3)')

        # To visualize pre-processed training images and their bboxes
        if not os.path.exists(os.path.join(save_root, 'visualization')):
            process = subprocess.Popen(['python', 'tools/misc/browse_dataset.py',
                                        config_file,
                                        '--output-dir', os.path.join(save_root, 'visualization'),
                                        '--not-show'])
            process.wait()

            # for idx, img_path in enumerate(natsorted(glob.glob(os.path.join(save_root, 'visualization', '*.png')))):
            #     if idx == 30:
            #         break

            #     fig = plt.figure(figsize=(10, 18))
            #     img = mpimg.imread(img_path)
            #     imgplot = plt.imshow(img)
            #     plt.show()
            #     plt.title(os.path.basename(img_path))
            #     plt.tight_layout()
            #     plt.axis('off')
            #     writer.add_figure('Train Data',
            #                     fig, global_step=idx)
            #     plt.close()

            visualize_img_tensorboard(save_path=os.path.join(save_root, 'visualization'),
                                      log_title='Train Data',
                                      num_imgs=30)




        # Plot FROC curve and PR curve
        # if not os.path.exists(os.path.join(save_root, 'result.bbox.json')):
        #     os.chdir(save_root)
        #     sys.path.append(save_root)
        #     # from faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm import DDSM_TEST_ANNOTATION
        #     config_mod = importlib.import_module(os.path.basename(config_file.split('.')[0]))

        #     os.chdir(eval_src_root)
        #     process = subprocess.Popen(['python', 'plot_eval_curve.py',
        #                                 '-gt', config_mod.TEST_ANNOTATION,
        #                                 '-p', os.path.join(save_root, 'result.bbox.json'),
        #                                 '--log_title', 'test',
        #                                 '-bb', 'all',
        #                                 '-s', os.path.join(save_root, 'curves')])
        #     process.wait()


        if not os.path.exists(os.path.join(save_root, 'train_result.bbox.json')):
            os.chdir(save_root)
            sys.path.append(save_root)
            config_mod = importlib.import_module(os.path.basename(config_file.split('.')[0]))


            os.chdir(mmdet_root)
            process = subprocess.Popen(['sh',  'tools/dist_test.sh',
                                        config_file,
                                        best_ckpt,
                                        '4',
                                        '--cfg-options',
                                        'data.test.img_prefix=' + config_mod.TRAIN_DATASET,
                                        'data.test.ann_file=' + config_mod.TRAIN_ANNOTATION,
                                        '--format-only', '--eval-options',
                                        'jsonfile_prefix=' + os.path.join(save_root, 'train_result')])
            process.wait()

            os.chdir(eval_src_root)
            process = subprocess.Popen(['python', 'plot_eval_curve.py',
                                        '-gt', config_mod.TRAIN_ANNOTATION,
                                        '-p', os.path.join(save_root, 'train_result.bbox.json'),
                                        '--log_title', 'train',
                                        '-bb', 'all',
                                        '-s', os.path.join(save_root, 'curves')])
            process.wait()

            for curve_img in glob.glob(os.path.join(save_root, 'curves', '*.png')):
                visualize_curve_tensorboard(curve_img)
