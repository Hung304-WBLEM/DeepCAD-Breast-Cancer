import numpy as np
import os
import glob
import random
import torch
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from torchvision import transforms
import torch.nn.functional as F


def compute_classes_weights(data_root, classes_names, combined_classes_names=None):
    num_classes = len(classes_names)

    weights = np.zeros(num_classes)

    for idx, class_name in enumerate(classes_names):
        weights[idx] = len(
            glob.glob(os.path.join(data_root, class_name, '*.png')))

    if combined_classes_names is not None:
        for combined_class_name in combined_classes_names:
            for label in combined_class_name.split('-'):
                idx = np.where(classes_names == label)

                weights[idx] += len(
                    glob.glob(os.path.join(data_root, combined_class_name, '*.png')))

    total_samples = np.sum(weights)

    weights = (1/weights) * total_samples

    return weights


def compute_classes_weights_mass_calc(mass_root, calc_root, classes_names):
    num_classes = len(classes_names)

    weights = np.zeros(num_classes)

    for idx, class_name in enumerate(classes_names):
        weights[idx] = len(glob.glob(os.path.join(mass_root, class_name, '*.png')) +
                           glob.glob(os.path.join(calc_root, class_name, '*.png')))

    total_samples = np.sum(weights)

    weights = (1/weights) * total_samples / num_classes

    return weights


def compute_classes_weights_mass_calc_pathology_4class(mass_root, calc_root, classes_names):
    num_classes = len(classes_names)

    weights = np.zeros(num_classes)

    for idx, class_name in enumerate(classes_names):
        pathology, lesion_type = class_name.split('_')

        if lesion_type == 'MASS':
            weights[idx] = len(glob.glob(os.path.join(mass_root, pathology, '*.png')))
        elif lesion_type == 'CALC':
            weights[idx] = len(glob.glob(os.path.join(calc_root, pathology, '*.png')))

    total_samples = np.sum(weights)

    weights = (1/weights) * total_samples / num_classes

    return weights


def compute_classes_weights_within_batch(batch_labels):
    '''
    Params:
    batch_labels - torch array of label. e.g.: torch([0, 1, 1, 2, 3, 4])
    '''
    classes, counts = torch.unique(batch_labels, return_counts=True)

    batch_weights = torch.zeros(batch_labels.shape[0])
    weights = torch.zeros(classes.shape[0])

    classes_map = dict()
    for idx, class_id in enumerate(classes):
        classes_map[class_id.item()] = idx

        weights[idx] = torch.sum(batch_labels == class_id)

    weights = (1/weights) * batch_labels.shape[0] / classes.shape[0]

    batch_weights = [weights[classes_map[label.item()]] for label in batch_labels]

    # Log info
    log_batch_weights = [el.item() for el in batch_weights]
    log_batch_labels = batch_labels.cpu().numpy().tolist()

    log_info = list(zip(log_batch_labels, log_batch_weights))

    # print('[+] Compute Classes Weights within Batch')
    # print('(Label, Weight): Frequency -', Counter(el for el in log_info))

    return batch_weights


def compute_classes_weights_mass_calc_pathology_5class(mass_root, calc_root, bg_root, classes_names):
    num_classes = len(classes_names)

    weights = np.zeros(num_classes)

    for idx, class_name in enumerate(classes_names):
        if class_name == 'BG':
            weights[idx] = len(glob.glob(os.path.join(bg_root, '*.png')))
        else:
            pathology, lesion_type = class_name.split('_')

            if lesion_type == 'MASS':
                weights[idx] = len(glob.glob(os.path.join(mass_root, pathology, '*.png')))
            elif lesion_type == 'CALC':
                weights[idx] = len(glob.glob(os.path.join(calc_root, pathology, '*.png')))

    total_samples = np.sum(weights)

    weights = (1/weights) * total_samples / num_classes

    return weights


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
