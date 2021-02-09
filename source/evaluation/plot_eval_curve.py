import os
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

from utils.fileio import json
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

    left, top, right, bottom, _ = pred_bbox
    center_x = (left+right) / 2
    center_y = (top + bottom) / 2

    if center_x > gt_bbox[0] and center_x < gt_bbox[2] and center_y > gt_bbox[1] and center_y < gt_bbox[3]:
        return True
    return False


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

    true_pos = 0
    false_pos = 0

    pred_bboxes = [(img_id, pred_bbox) for img_id, img_pred_bboxes in enumerate(
        pred_bboxes_list) for pred_bbox in img_pred_bboxes]
    pred_bboxes = sorted(
        pred_bboxes, key=lambda x: x[1][4], reverse=True)
    print('#Pred boxes:', len(pred_bboxes))

    matched = [[False for gt_bbox in gt_bboxes]
               for gt_bboxes in gt_bboxes_list]
    total_pos = sum([len(gt_bboxes) for gt_bboxes in gt_bboxes_list])
    print('#Gt boxes:', total_pos)

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

        precision_values.append(true_pos/(true_pos+false_pos))
        recall_values.append(true_pos/(total_pos))

    for idx in range(0, len(precision_values)-1):
        precision_values[idx] = max(precision_values[idx+1:])

    ap = sum([(recall_values[r] - recall_values[r-1])*precision_values[r]
              for r in range(1, len(precision_values))])
    ap = round(ap, 2)
    print('Average Precision:', ap)

    return precision_values, recall_values, ap


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

    true_pos = 0
    false_pos = 0

    pred_bboxes = [(img_id, pred_bbox) for img_id, img_pred_bboxes in enumerate(
        pred_bboxes_list) for pred_bbox in img_pred_bboxes]
    pred_bboxes = sorted(
        pred_bboxes, key=lambda x: x[1][4], reverse=True)
    print('#Pred boxes:', len(pred_bboxes))

    matched = [[False for gt_bbox in gt_bboxes]
               for gt_bboxes in gt_bboxes_list]
    total_pos = sum([len(gt_bboxes) for gt_bboxes in gt_bboxes_list])
    print('#Gt boxes:', total_pos)

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

        precision_values.append(true_pos/(true_pos+false_pos))
        recall_values.append(true_pos/(total_pos))

    for idx in range(0, len(precision_values)-1):
        precision_values[idx] = max(precision_values[idx+1:])

    ap = sum([(recall_values[r] - recall_values[r-1])*precision_values[r]
              for r in range(1, len(precision_values))])
    ap = round(ap, 2)
    print('Average Precision:', ap)

    return precision_values, recall_values, ap


def plot_prec_rec_true_pos_metric(_save_path, _gt_bboxes_json, _pred_bboxes_json, _category_id, _bbox_select):
    gt_bboxes_list, pred_bboxes_list = \
        get_bboxes_lists(gt_bboxes_json=_gt_bboxes_json,
                         pred_bboxes_json=_pred_bboxes_json,
                         category_id=_category_id, bbox_select=_bbox_select)

    iou75_prec, iou75_rec, iou75_ap = detection_prec_rec(
        gt_bboxes_list, pred_bboxes_list, iou_thres=0.75)
    iou50_prec, iou50_rec, iou50_ap = detection_prec_rec(
        gt_bboxes_list, pred_bboxes_list, iou_thres=0.5)
    iou25_prec, iou25_rec, iou25_ap = detection_prec_rec(
        gt_bboxes_list, pred_bboxes_list, iou_thres=0.25)
    center_prec, center_rec, center_ap = detection_loose_prec_rec(
        gt_bboxes_list, pred_bboxes_list)

    fig = plt.figure()
    plt.plot(iou75_rec, iou75_prec,
             label=f"IoU=0.75 (AUC={iou75_ap})", color="blue")
    plt.plot(iou50_rec, iou50_prec,
             label=f"IoU=0.5 (AUC={iou50_ap})", color="orange")
    plt.plot(iou25_rec, iou25_prec,
             label=f"IoU=0.25 (AUC={iou25_ap})", color="green")
    plt.plot(center_rec, center_prec,
             label=f"center (AUC={center_ap})", color="red")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)

    plt.savefig(_save_path)
    plt.close()

    return iou50_ap, iou25_ap, center_ap


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

    print('#test images:', len(gt_json['images']))
    for image in gt_json['images']:
        image_id = image['id']

        gt_bboxes = []
        for ann in gt_json['annotations']:
            if ann['category_id'] != category_id:
                continue
            if ann['image_id'] == image_id:
                x1, y1, w, h = ann['bbox']
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                gt_bboxes.append((x1, y1, x2, y2))
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
                    selected_bb = (x1, y1, x2, y2, s)
                elif bbox_select == 'all':
                    pred_bboxes.append((x1, y1, x2, y2, s))

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
    # parser.add_argument("-c", "--category_id", type=int,
    #                     help="the ID of class you want to evaluate")
    parser.add_argument("-bb", "--bbox_select", choices={"all", "opi"},
                        help="either `all` or `opi` for choose positive bounding boxes during evaluation")
    parser.add_argument("-s", "--save_path", help="choose path to save figure")

    args = parser.parse_args()
    vars_dict = vars(args)

    classes = {'malignant': 0, 'benign': 1}

    ap_values = []
    for class_name, class_id in classes.items():
        iou50_ap, iou25_ap, center_ap = plot_prec_rec_true_pos_metric(os.path.join(args.save_path, f'plot_cmp_tp_metrics_{class_name}.png'), args.gt_bboxes_json,
                                                                      args.pred_bboxes_json, class_id, args.bbox_select)
        ap_values.append([class_name, class_id, iou50_ap, iou25_ap, center_ap])

    maps = np.mean(np.array([ret[2:] for ret in ap_values]), axis=1)
    print(ap_values)
    print(maps)
