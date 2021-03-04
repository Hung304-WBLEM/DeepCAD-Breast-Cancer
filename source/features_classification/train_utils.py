import numpy as np
import os
import glob
import random
import torch
import matplotlib
import matplotlib.pyplot as plt

def compute_classes_weights(data_root, classes_names):
    num_classes = len(classes_names)

    weights = np.zeros(num_classes)
    for idx, class_name in enumerate(classes_names):
        weights[idx] = len(
            glob.glob(os.path.join(data_root, class_name, '*.png')))

    total_samples = np.sum(weights)

    weights = (1/weights) * total_samples / num_classes

    return weights


def compute_classes_weights_mass_calc_pathology(mass_root, calc_root, classes_names):
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


def plot_train_val_loss(num_epochs, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, save_path):
    fig = plt.figure()
    plt.plot(range(num_epochs), train_loss_hist, label='train loss')
    plt.plot(range(num_epochs), val_loss_hist, label='val loss')
    plt.xlabel('#epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    fig = plt.figure()
    plt.plot(range(num_epochs), train_acc_hist, label='train accuracy')
    plt.plot(range(num_epochs), val_acc_hist, label='val accuracy')
    plt.xlabel('#epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc_plot.png'))
    plt.close()

