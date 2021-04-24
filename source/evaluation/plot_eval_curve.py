import os
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

from utilities.fileio import json
from sklearn.metrics import auc
from torch.utils.tensorboard import SummaryWriter
matplotlib.use('Agg')


def IoU(bbox_a, bbox_b):
    '''
    bbox_a - (left, top, right, bottom)
    bbox_b - (left, top, right, bottom)
    '''

    intersec_left = max(bbox_a[0], bbox_b[0])
    intersec_top = max(bbox_a[1], bbox_b[1])
    intersec_right = min(bbox_a[2], bbox_b[2])
    intersec_bottom = min(bbox_a[3], bbox_b[3])

    intersec_area = max(intersec_right - intersec_left + 1, 0) * \
        max(intersec_bottom - intersec_top + 1, 0)

    bbox_a_area = (bbox_a[2] - bbox_a[0]+1) * (bbox_a[3]-bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0]+1) * (bbox_b[3]-bbox_b[1] + 1)
    union_area = bbox_a_area + bbox_b_area - intersec_area

    return intersec_area / union_area


def isCenterInGTbbox(pred_bbox, gt_bbox):
    '''
    pred_bbox - (left, top, right, bottom, score)
    gt_bbox - (left, top, right, bottom)
    '''

    left, top, right, bottom, _, _ = pred_bbox
    center_x = (left+right) / 2
    center_y = (top + bottom) / 2

    if center_x > gt_bbox[0] and center_x < gt_bbox[2] and center_y > gt_bbox[1] and center_y < gt_bbox[3]:
        return True
    return False


def all_classes_detection_prec_rec(gt_categories, _gt_bboxes_json, _pred_bboxes_json, _bbox_select, _iou_thres):
    def detection_prec_rec(gt_bboxes_list, pred_bboxes_list, iou_thres):
        ''' Get precision and recall values
        pred_bboxes_list - list of sublists of predicted bounding boxes.
                        Each sublist represents all detected bounding
                        boxes of one specific image (bbox_format: 
                        (left, top, right, bottom, score))
        gt_bboxes_list - list of sublists of ground-truth bounding boxes.
                        Each sublist represents all ground-truth bounding
                        boxes of one specific image (bbox_format: 
                        (left, top, right, bottom))
        '''

        precision_values = []
        recall_values = []
        false_pos_per_img_values = []
        images_set = set()
        img_filenames = []

        true_pos = 0
        false_pos = 0

        pred_bboxes = [(img_id, pred_bbox) for img_id, img_pred_bboxes in enumerate(
            pred_bboxes_list) for pred_bbox in img_pred_bboxes]
        pred_bboxes = sorted(
            pred_bboxes, key=lambda x: x[1][4], reverse=True)
        # print('#Pred boxes:', len(pred_bboxes))

        matched = [[False for gt_bbox in gt_bboxes]
                   for gt_bboxes in gt_bboxes_list]
        total_pos = sum([len(gt_bboxes) for gt_bboxes in gt_bboxes_list])
        # print('#Gt boxes:', total_pos)

        for img_id, pred_bbox in pred_bboxes:

            max_iou = -1
            selected_gt_bbox_id = -1

            for gt_bbox_id, gt_bbox in enumerate(gt_bboxes_list[img_id]):
                if matched[img_id][gt_bbox_id] is True:
                    continue

                iou = IoU(pred_bbox, gt_bbox)
                if iou >= iou_thres and iou > max_iou:
                    max_iou = iou
                    selected_gt_bbox_id = gt_bbox_id

            if selected_gt_bbox_id != -1:
                matched[img_id][selected_gt_bbox_id] = True
                true_pos += 1
            else:
                false_pos += 1

            images_set.add(img_id)

            precision_values.append(true_pos/(true_pos+false_pos))
            recall_values.append(true_pos/(total_pos))
            false_pos_per_img_values.append(false_pos/len(images_set))
            img_filenames.append(pred_bbox[5])

        # Compute Average Precision
        ap = sum([(recall_values[r] - recall_values[r-1])*precision_values[r]
                  for r in range(1, len(precision_values))])
        ap = round(ap, 2)
        # print('Average Precision:', ap)

        for idx in range(0, len(precision_values)-1):
            precision_values[idx] = max(precision_values[idx:])
            false_pos_per_img_values[idx] = min(
                false_pos_per_img_values[idx:])

        return precision_values, recall_values, false_pos_per_img_values, ap, img_filenames

    # Evaluation for all classes
    eval_log = dict()
    eval_plot = dict()
    aps = []
    for class_info in gt_categories:
        _category_id = class_info['id']

        gt_bboxes_list, pred_bboxes_list  = \
            get_bboxes_lists(gt_bboxes_json=_gt_bboxes_json,
                             pred_bboxes_json=_pred_bboxes_json,
                             category_id=_category_id, bbox_select=_bbox_select)
        p_vals, r_vals, fp_img_vals, ap, img_filenames = detection_prec_rec(
            gt_bboxes_list, pred_bboxes_list, _iou_thres)
        eval_log[_category_id] = {'AP': ap}
        eval_plot[_category_id] = {'precisions': p_vals, 'recalls': r_vals,
                                   'false_positives_per_image': fp_img_vals, 'auc': ap,
                                   'img_filenames': img_filenames}
        aps.append(ap)

    eval_log['mAP'] = sum(aps)/len(aps)
    return eval_log, eval_plot


def all_classes_detection_loose_prec_rec(gt_categories, _gt_bboxes_json, _pred_bboxes_json, _bbox_select):
    def detection_loose_prec_rec(gt_bboxes_list, pred_bboxes_list):
        ''' Plot precision-recall curve using the center metric, i.e.,
        if the center of the predicted box is in the ground-truth box, is
        will be determined as true positive alarm.

        pred_bboxes_list - list of sublists of predicted bounding boxes.
                        Each sublist represents all detected bounding
                        boxes of one specific image (bbox_format: 
                        (left, top, right, bottom, score))
        gt_bboxes_list - list of sublists of ground-truth bounding boxes.
                        Each sublist represents all ground-truth bounding
                        boxes of one specific image (bbox_format: 
                        (left, top, right, bottom))
        '''

        precision_values = []
        recall_values = []
        false_pos_per_img_values = []
        images_set = set()
        img_filenames = []

        true_pos = 0
        false_pos = 0

        pred_bboxes = [(img_id, pred_bbox) for img_id, img_pred_bboxes in enumerate(
            pred_bboxes_list) for pred_bbox in img_pred_bboxes]
        pred_bboxes = sorted(
            pred_bboxes, key=lambda x: x[1][4], reverse=True)
        # print('#Pred boxes:', len(pred_bboxes))

        matched = [[False for gt_bbox in gt_bboxes]
                   for gt_bboxes in gt_bboxes_list]
        total_pos = sum([len(gt_bboxes) for gt_bboxes in gt_bboxes_list])
        # print('#Gt boxes:', total_pos)

        for img_id, pred_bbox in pred_bboxes:

            selected_gt_bbox_id = -1

            for gt_bbox_id, gt_bbox in enumerate(gt_bboxes_list[img_id]):
                if matched[img_id][gt_bbox_id] is True:
                    continue

                if isCenterInGTbbox(pred_bbox, gt_bbox):
                    selected_gt_bbox_id = gt_bbox_id

            if selected_gt_bbox_id != -1:
                matched[img_id][selected_gt_bbox_id] = True
                true_pos += 1
            else:
                false_pos += 1

            images_set.add(img_id)

            precision_values.append(true_pos/(true_pos+false_pos))
            recall_values.append(true_pos/(total_pos))
            false_pos_per_img_values.append(false_pos/len(images_set))
            img_filenames.append(pred_bbox[5])

        # Compute Average Precision
        ap = sum([(recall_values[r] - recall_values[r-1])*precision_values[r]
                  for r in range(1, len(precision_values))])
        ap = round(ap, 2)
        # print('Average Precision:', ap)

        for idx in range(0, len(precision_values)-1):
            precision_values[idx] = max(precision_values[idx:])
            false_pos_per_img_values[idx] = min(
                false_pos_per_img_values[idx:])

        return precision_values, recall_values, false_pos_per_img_values, ap, img_filenames

    # Evaluation for all classes
    eval_log = dict()
    eval_plot = dict()
    aps = []
    for class_info in gt_categories:
        _category_id = class_info['id']

        gt_bboxes_list, pred_bboxes_list = \
            get_bboxes_lists(gt_bboxes_json=_gt_bboxes_json,
                             pred_bboxes_json=_pred_bboxes_json,
                             category_id=_category_id, bbox_select=_bbox_select)
        p_vals, r_vals, fp_img_vals, ap, img_filenames = detection_loose_prec_rec(
            gt_bboxes_list, pred_bboxes_list)
        eval_log[_category_id] = {'AP': ap}
        eval_plot[_category_id] = {'precisions': p_vals, 'recalls': r_vals,
                                   'false_positives_per_image': fp_img_vals, 'auc': ap,
                                   'img_filenames': img_filenames}
        aps.append(ap)

    eval_log['mAP'] = sum(aps)/len(aps)
    return eval_log, eval_plot


def plot_pr_curve_true_pos_metric(_save_path, _bbox_select, gt_categories, iou75_eval_plot, iou50_eval_plot, iou25_eval_plot, center_eval_plot, fig_only=False):
    writer = SummaryWriter(os.path.join(_save_path, 'tensorboard_logs'))

    for class_info in gt_categories:
        class_id = class_info['id']
        class_name = class_info['name']

        iou75_prec, iou75_rec, iou75_fp_img, iou75_ap = \
            iou75_eval_plot[class_id]['precisions'], \
            iou75_eval_plot[class_id]['recalls'], \
            iou75_eval_plot[class_id]['false_positives_per_image'], \
            iou75_eval_plot[class_id]['auc']
        iou50_prec, iou50_rec, iou50_fp_img, iou50_ap = \
            iou50_eval_plot[class_id]['precisions'], \
            iou50_eval_plot[class_id]['recalls'], \
            iou50_eval_plot[class_id]['false_positives_per_image'], \
            iou50_eval_plot[class_id]['auc']
        iou25_prec, iou25_rec, iou25_fp_img, iou25_ap = \
            iou25_eval_plot[class_id]['precisions'], \
            iou25_eval_plot[class_id]['recalls'], \
            iou25_eval_plot[class_id]['false_positives_per_image'], \
            iou25_eval_plot[class_id]['auc']
        center_prec, center_rec, center_fp_img, center_ap = \
            center_eval_plot[class_id]['precisions'], \
            center_eval_plot[class_id]['recalls'], \
            center_eval_plot[class_id]['false_positives_per_image'], \
            center_eval_plot[class_id]['auc']

        # Draw Precision-Recall curves
        fig = plt.figure()
        plt.plot(iou75_rec, iou75_prec,
                 label=f"IoU=0.75 (AUC={iou75_ap})")
        plt.plot(iou50_rec, iou50_prec,
                 label=f"IoU=0.5 (AUC={iou50_ap})")
        plt.plot(iou25_rec, iou25_prec,
                 label=f"IoU=0.25 (AUC={iou25_ap})")
        plt.plot(center_rec, center_prec,
                 label=f"center (AUC={center_ap})")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        plt.title(f'PR curves for class: {class_name}')

        if _save_path is not None:
            plt.savefig(os.path.join(
                _save_path, f'precision-recall_curve_{class_name}_{_bbox_select}_cmp_tp_metrics.png'))

        if fig_only:
            writer.add_figure(f'test pr curve - {class_name}', fig, global_step=None)

        plt.close()

    


def plot_froc_curve_true_pos_metric(_save_path, _bbox_select, gt_categories, iou75_eval_plot, iou50_eval_plot, iou25_eval_plot, center_eval_plot, fig_only=False):
    writer = SummaryWriter(os.path.join(_save_path, 'tensorboard_logs'))

    for class_info in gt_categories:
        class_id = class_info['id']
        class_name = class_info['name']

        iou75_prec, iou75_rec, iou75_fp_img, iou75_ap = \
            iou75_eval_plot[class_id]['precisions'], \
            iou75_eval_plot[class_id]['recalls'], \
            iou75_eval_plot[class_id]['false_positives_per_image'], \
            iou75_eval_plot[class_id]['auc']
        iou50_prec, iou50_rec, iou50_fp_img, iou50_ap = \
            iou50_eval_plot[class_id]['precisions'], \
            iou50_eval_plot[class_id]['recalls'], \
            iou50_eval_plot[class_id]['false_positives_per_image'], \
            iou50_eval_plot[class_id]['auc']
        iou25_prec, iou25_rec, iou25_fp_img, iou25_ap = \
            iou25_eval_plot[class_id]['precisions'], \
            iou25_eval_plot[class_id]['recalls'], \
            iou25_eval_plot[class_id]['false_positives_per_image'], \
            iou25_eval_plot[class_id]['auc']
        center_prec, center_rec, center_fp_img, center_ap = \
            center_eval_plot[class_id]['precisions'], \
            center_eval_plot[class_id]['recalls'], \
            center_eval_plot[class_id]['false_positives_per_image'], \
            center_eval_plot[class_id]['auc']


        # Draw FROC curves
        fig = plt.figure()
        plt.plot(iou75_fp_img, iou75_rec,
                 label=f"IoU=0.75")
        plt.plot(iou50_fp_img, iou50_rec,
                 label=f"IoU=0.5")
        plt.plot(iou25_fp_img, iou25_rec,
                 label=f"IoU=0.25")
        plt.plot(center_fp_img, center_rec,
                 label=f"center")

        plt.xlabel('Number of false positive marks per image')
        plt.ylabel('Sensitivity')
        plt.legend()
        plt.grid(True)
        plt.title(f'FROC curves for class: {class_name}')

        if _save_path is not None:
            plt.savefig(os.path.join(
                _save_path, f'froc_curve_{class_name}_{_bbox_select}_cmp_tp_metrics.png'))

        if fig_only:
            writer.add_figure(f'test froc curve - {class_name}', fig, global_step=None)

        plt.close()


def get_bboxes_lists(gt_bboxes_json, pred_bboxes_json, category_id, bbox_select='all'):
    ''' Load ground-truth and predicted bounding boxes data of specific
    category ID for evaluation

    Args:
    gt_bboxes_json (str): path to the ground-truth json file
    pred_bboxes_json (str): path to the ground-truth json file
    category_id (int): id of the class you want to evaluate
    bbox_select (str: 'all' | 'opi'): if this is set to 'all', all predicted bounding boxes will
    be selected for evaluation. 'opi' is one-per-image, this will only choose the bounding box
    with the highest score for each image. 

    Returns:
    gt_bboxes_list (list) - list of ground-truth boxes
    pred_bboxes_list (list) - list of predicted boxes
    '''

    gt_json = json.read(gt_bboxes_json)
    pred_json = json.read(pred_bboxes_json)

    gt_bboxes_list = []
    pred_bboxes_list = []

    # print('#test images:', len(gt_json['images']))
    for image in gt_json['images']:
        image_id = image['id']
        filename = image['file_name']

        gt_bboxes = []
        for ann in gt_json['annotations']:
            if ann['category_id'] != category_id:
                continue
            if ann['image_id'] == image_id:
                x1, y1, w, h = ann['bbox']
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                gt_bboxes.append((x1, y1, x2, y2, filename))
        gt_bboxes_list.append(gt_bboxes)

        pred_bboxes = []
        if bbox_select == 'opi':
            best_score = 0
            selected_bb = None

        for pred in pred_json:
            if pred['category_id'] != category_id:
                continue
            if pred['image_id'] == image_id:
                x1, y1, w, h = pred['bbox']
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                s = pred['score']

                if bbox_select == 'opi' and s > best_score:
                    best_score = s
                    selected_bb = (x1, y1, x2, y2, s, filename)
                elif bbox_select == 'all':
                    pred_bboxes.append((x1, y1, x2, y2, s, filename))

        if bbox_select == 'opi' and selected_bb is not None:
            pred_bboxes.append(selected_bb)

        pred_bboxes_list.append(pred_bboxes)

    return gt_bboxes_list, pred_bboxes_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gt", "--gt_bboxes_json", help="path to ground-truth boxes json file")
    parser.add_argument("-p", "--pred_bboxes_json",
                        help="path to predicted boxes json file")
    parser.add_argument("-bb", "--bbox_select", choices={"all", "opi"},
                        help="either `all` or `opi` for choose positive bounding boxes during evaluation")
    parser.add_argument("-s", "--save_path", help="choose path to save figure")

    args = parser.parse_args()
    vars_dict = vars(args)

    gt_json = json.read(args.gt_bboxes_json)
    # print('Ground-Truth Categories:', gt_json['categories'])
    gt_categories = gt_json['categories']

    iou75_eval_log, iou75_eval_plot = all_classes_detection_prec_rec(
        gt_categories,
        args.gt_bboxes_json,
        args.pred_bboxes_json,
        args.bbox_select,
        _iou_thres=0.75)
    iou50_eval_log, iou50_eval_plot = all_classes_detection_prec_rec(
        gt_categories,
        args.gt_bboxes_json,
        args.pred_bboxes_json,
        args.bbox_select,
        _iou_thres=0.5)
    iou25_eval_log, iou25_eval_plot = all_classes_detection_prec_rec(
        gt_categories,
        args.gt_bboxes_json,
        args.pred_bboxes_json,
        args.bbox_select,
        _iou_thres=0.25)
    print(list(zip(iou25_eval_plot[1]['recalls'], iou25_eval_plot[1]['img_filenames'])))
    center_eval_log, center_eval_plot = all_classes_detection_loose_prec_rec(
        gt_categories,
        args.gt_bboxes_json,
        args.pred_bboxes_json,
        args.bbox_select)

    # Plot PR curves
    plot_pr_curve_true_pos_metric(args.save_path,
                                  args.bbox_select,
                                  gt_categories,
                                  iou75_eval_plot,
                                  iou50_eval_plot,
                                  iou25_eval_plot,
                                  center_eval_plot,
                                  fig_only=True)

    # Plot FROC curves
    plot_froc_curve_true_pos_metric(args.save_path,
                                  args.bbox_select,
                                  gt_categories,
                                  iou75_eval_plot,
                                  iou50_eval_plot,
                                  iou25_eval_plot,
                                  center_eval_plot,
                                  fig_only=True)

    json.write({'categories': gt_categories,
                'center': center_eval_log,
                'iou25': iou25_eval_log,
                'iou50': iou50_eval_log,
                'iou75': iou75_eval_log}, os.path.join(args.save_path, 'eval_log.json'))
