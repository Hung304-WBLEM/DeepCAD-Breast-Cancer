import numpy as np
import os
import glob
import random
import torch
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
import torch.nn.functional as F


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


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    if isinstance(preds.tolist(), int):
        preds = [preds]

    return preds, \
        [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)], \
        [F.softmax(el, dim=0).cpu().detach().numpy() for el in output]


def images_to_probs_pathology(net, images, input_vectors):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images, input_vectors, training=False)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_score_bars(all_classes_prob, classes):
    labels = ['classes']
    x = np.arange(len(labels))
    width = 0.05

    num_classes = len(classes)

    for idx, prob in enumerate(all_classes_prob.tolist()):
        class_score = [prob]
        rect = plt.bar(idx*width + width/2, class_score, width, label=classes[idx])

        plt.yticks([0, 0.25, 0.5, 0.75, 1])

    plt.legend(prop={'size': 6})


def plot_classes_preds(net, images, labels, classes, num_images):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs, all_classes_probs = images_to_probs(net, images)
    # print(all_classes_probs)
    # while True:
    #     continue
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 40))
    # for idx in np.arange(num_images):
    for idx in range(0, num_images):
        import math
        ax = fig.add_subplot(math.ceil(num_images/4)*2, 4, 2*idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)

        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                     color=("green" if preds[idx]==labels[idx].item() else "red"),
                     fontsize=10)

        ax = fig.add_subplot(math.ceil(num_images/4)*2, 4, 2*idx+2, xticks=[], yticks=[])
        show_score_bars(all_classes_probs[idx], classes)

    return fig


def plot_classes_preds_pathology(net, images, input_vectors, labels, classes, num_images):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs_pathology(net, images, input_vectors)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 15))
    for idx in np.arange(num_images):
        import math
        ax = fig.add_subplot(math.ceil(num_images/4), 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                     color=("green" if preds[idx]==labels[idx].item() else "red"),
                     fontsize=10)
    return fig


def add_pr_curve_tensorboard(writer, classes, class_index, test_labels, test_probs, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_labels = test_labels == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_labels,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()
