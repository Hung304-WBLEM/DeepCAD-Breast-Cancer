import pickle
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from scipy import interp


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.RdPu):
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    classes = list(map(lambda x: x.lower(), classes))

    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    norm_cm = np.nan_to_num(norm_cm)

    print(cm)
    print(norm_cm)
    plt.imshow(norm_cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    thresh = (norm_cm.max() + norm_cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
        #          color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i, '{0:.2f}\n({1:d})'.format(norm_cm[i,j], cm[i,j]), verticalalignment="center", horizontalalignment="center",
                 color="white" if norm_cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return cm


def evalplot_confusion_matrix(y_true, y_pred, classes, save_root=None, fig_only=False):
    classes = list(map(lambda x: x.lower(), classes))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # plt.figure(figsize=(10, 10))
    # plot_confusion_matrix(cm, classes)
    # if save_root is not None:
    #     plt.savefig(os.path.join(save_root, 'confusion_matrix.png'))
    # plt.close()

    # Normalized Confusion Matrix
    # plt.figure(figsize=(10, 10))
    # norm_cm = plot_confusion_matrix(cm, classes, normalize=True)
    # if save_root is not None:
    #     plt.savefig(os.path.join(save_root, 'norm_confusion_matrix.png'))
    # plt.close()
    label_freq = np.unique(y_true).tolist()
    pred_freq = np.unique(y_pred).tolist()
    avail_classes = []
    for class_id, class_name in enumerate(classes):
        if class_id not in label_freq and class_id not in pred_freq:
            continue
        avail_classes.append(class_name)

    fig = plt.figure(figsize=(10, 10))
    cm = plot_confusion_matrix(cm, avail_classes)
    if save_root is not None:
        plt.savefig(os.path.join(save_root, 'confusion_matrix.png'))

    if fig_only:
        return fig

    plt.close()

    return cm


def plot_precision_recall_curve(binarized_y_true, y_proba_pred, class_name, color):
    precisions, recalls, thresholds = precision_recall_curve(
        binarized_y_true, y_proba_pred)
    ap = average_precision_score(binarized_y_true, y_proba_pred)
    ap = round(ap, 2)

    precisions = precisions[:-1].tolist()
    precisions.reverse()
    recalls = recalls[:-1].tolist()
    recalls.reverse()
    thresholds = thresholds[:-1].tolist()
    thresholds.reverse()

    interpolated_precisions = [0] * len(precisions)
    for idx in range(0, len(precisions) - 1):
        interpolated_precisions[idx] = max(precisions[idx:])

    plt.plot(recalls[:], interpolated_precisions[:],
             label=f"{class_name} (AP={ap})", color=color, linewidth=2)

    return precisions, recalls, ap


def evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes, save_root=None, fig_only=False, cmap=[plt.cm.tab20, plt.cm.tab20b]):
    classes = list(map(lambda x: x.lower(), classes))
    n_classes = len(classes)

    precisions_list = []
    recalls_list = []
    average_precisions_list = []

    if fig_only:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(8, 6))


    if binarized_y_true.shape[1] == 1:  # For binary clasification
        precisions, recalls, average_precisions = plot_precision_recall_curve(
            binarized_y_true, y_proba_pred[:, 1], classes[1])  # Assume class 1 is positive
        precisions_list.append(precisions)
        recalls_list.append(recalls)
        average_precisions_list.append(average_precisions)
    else:  # For multiclass classification
        colors = [cmap[0](np.linspace(0, 1, n_classes-n_classes//2)),
                  cmap[1](np.linspace(0, 1, n_classes//2))]

        for class_id, class_name in enumerate(classes):
            if np.sum(binarized_y_true[:, class_id]) > 0:
                precisions, recalls, average_precisions = plot_precision_recall_curve(
                    binarized_y_true[:, class_id], y_proba_pred[:, class_id], class_name,
                    color=colors[class_id%2][class_id//2])
                precisions_list.append(precisions)
                recalls_list.append(recalls)
                average_precisions_list.append(average_precisions)

    # Plot micro-average PR curve
    micro_precisions, micro_recalls, _ = precision_recall_curve(
        binarized_y_true.ravel(), y_proba_pred.ravel())
    micro_average_precision = round(average_precision_score(binarized_y_true,
                                                            y_proba_pred, average="micro"), 2)

    micro_precisions = micro_precisions[:-1].tolist()
    micro_precisions.reverse()
    micro_recalls = micro_recalls[:-1].tolist()
    micro_recalls.reverse()

    interpolated_micro_precisions = [0] * len(micro_precisions)
    for idx in range(0, len(micro_precisions) - 1):
        interpolated_micro_precisions[idx] = max(micro_precisions[idx:])
    plt.plot(micro_recalls[:], interpolated_micro_precisions[:],
             label=f"Micro-average (AP={micro_average_precision})", color='blue',
             linewidth=1, linestyle='dashed')

    # Calculate macro-average AP
    macro_average_precision = round(average_precision_score(binarized_y_true,
                                                            y_proba_pred, average="macro"), 2)
    # --------------------------------------------------------------------------

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.grid(True)
    plt.tight_layout()

    if save_root is not None:
        plt.savefig(os.path.join(save_root, 'pr_curve.png'))

    if fig_only:
        return fig

    plt.close()

    log_info = { 
        'micro_average_precision': micro_average_precision,
        'macro_average_precision': macro_average_precision,
        'average_precisions_list': average_precisions_list
    }

    return precisions_list, recalls_list, average_precisions_list, log_info


def plot_roc_curve(binarized_y_true, y_proba_pred, class_name, color):
    fpr, tpr, thresholds = roc_curve(binarized_y_true, y_proba_pred)
    auc = roc_auc_score(binarized_y_true, y_proba_pred)
    auc = round(auc, 2)

    plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc})", color=color, linewidth=2)

    return fpr, tpr, auc


def evalplot_roc_curve(binarized_y_true, y_proba_pred, classes, save_root=None, fig_only=False, cmap=[plt.cm.tab20, plt.cm.tab20b]):
    classes = list(map(lambda x: x.lower(), classes))
    n_classes = len(classes)

    fprs_list, tprs_list, aucs_list = [], [], []

    if fig_only:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(8, 6))
    if binarized_y_true.shape[1] == 1:  # For binary classification
        fpr, tpr, _auc = plot_roc_curve(
            binarized_y_true, y_proba_pred[:, 1], class_name=classes[1])
        fprs_list.append(fpr)
        tprs_list.append(tpr)
        aucs_list.append(_auc)
    else:  # For multiclass classification
        colors = [cmap[0](np.linspace(0, 1, n_classes-n_classes//2)),
                  cmap[1](np.linspace(0, 1, n_classes//2))]

        for class_id, class_name in enumerate(classes):
            if np.sum(binarized_y_true[:, class_id]) > 0:
                fpr, tpr, _auc = plot_roc_curve(
                    binarized_y_true[:, class_id], y_proba_pred[:, class_id], class_name,
                    color=colors[class_id%2][class_id//2])
                fprs_list.append(fpr)
                tprs_list.append(tpr)
                aucs_list.append(_auc)


    # --------------------------------------------------------------------------
    # plotting micro-average ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(binarized_y_true.ravel(), y_proba_pred.ravel())
    micro_roc_auc = round(auc(micro_fpr, micro_tpr), 2)
    plt.plot(micro_fpr, micro_tpr, label=f"Micro-average (AUC={micro_roc_auc})", color='blue',
             linewidth=1, linestyle='dashed')

    # plotting macro-average ROC curve
    all_fpr = []
    for idx in range(len(fprs_list)):
        all_fpr.append(fprs_list[idx])

    all_fpr = np.unique(np.concatenate(all_fpr))
    mean_tpr = np.zeros_like(all_fpr)

    for idx in range(len(fprs_list)):
        mean_tpr += interp(all_fpr, fprs_list[idx], tprs_list[idx])

    mean_tpr /= n_classes

    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_roc_auc = round(auc(macro_fpr, macro_tpr), 2)

    plt.plot(macro_fpr, macro_tpr, label=f"Macro-average (AUC={macro_roc_auc})", color='red',
             linewidth=1, linestyle='dashed')

    # --------------------------------------------------------------------------
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    if save_root is not None:
        plt.savefig(os.path.join(save_root, 'roc_curve.png'))

    if fig_only:
        return fig

    plt.close()

    log_info = {
        'micro_roc_auc': micro_roc_auc,
        'macro_roc_auc': macro_roc_auc,
        'aucs_list': aucs_list
    }
    return fprs_list, tprs_list, aucs_list, log_info


@torch.no_grad()
def eval_all(y_true, preds, classes, save_root, multilabel_mode, dataset):
    matplotlib.use('Agg')

    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])

    if not multilabel_mode:
        y_proba_pred = torch.softmax(torch.from_numpy(preds), dim=-1).detach().numpy()
        y_pred = y_proba_pred.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
    else:
        y_proba_pred_softmax = torch.softmax(torch.from_numpy(preds), dim=-1).detach().numpy()
        y_pred = y_proba_pred_softmax.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)

        y_proba_pred = torch.sigmoid(torch.from_numpy(preds)).detach().numpy()

    if hasattr(dataset, 'combined_classes'):
        all_classes = np.concatenate((classes, dataset.combined_classes))
    else:
        all_classes = classes

    cm = evalplot_confusion_matrix(y_true, y_pred, all_classes, save_root)
    precisions_list, recalls_list, average_precisions_list, pr_log_info = \
        evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes, save_root)
    fprs_list, tprs_list, aucs_list, roc_log_info = \
        evalplot_roc_curve(binarized_y_true, y_proba_pred, classes, save_root)

    plot_data = {'accuracy': acc,
                 'confusion_matrix': cm,

                 'classes_precisions_list': precisions_list,
                 'classes_recalls_list': recalls_list,
                 'classes_average_precisions_list': average_precisions_list,
                 'macro_average_precision': pr_log_info['macro_average_precision'],
                 'micro_average_precision': pr_log_info['micro_average_precision'],

                 'classes_fprs_list': fprs_list,
                 'classes_tprs_list': tprs_list,
                 'classes_aucs_list': aucs_list,
                 'macro_roc_auc': roc_log_info['macro_roc_auc'],
                 'micro_roc_auc': roc_log_info['micro_roc_auc'],

                 'classes': classes,
                 'all_classes': all_classes}

    with open(os.path.join(save_root, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)

    return acc, \
        pr_log_info['macro_average_precision'], pr_log_info['micro_average_precision'], \
        pr_log_info['average_precisions_list'], \
        roc_log_info['macro_roc_auc'], roc_log_info['micro_roc_auc'], \
        roc_log_info['aucs_list']


def plot_learning_curve(save_root, metric, dataset, classes, *result_paths):
    '''Plot the learning curve for different experimental results directories
    
    Args:
    metric(str) - metric that you want to plot
    dataset(str) - dataset that you want to plot, this is to get information
    about classes of this dataset
    classes - list of class names
    *result_paths - list of tuples, each tuple contains the path and the corresponding data percentage 
    '''

    if metric == 'classes_aucs_list':
        ylabel = 'AUC_ROC'

    fig = plt.figure(figsize=(10, 10))

    
    all_result = []
    all_percents = []
    for path, percent in result_paths:
        with open(os.path.join(path, 'plot_data.pkl'), 'rb') as f:
            result = pickle.load(f)

        classes_result = result[metric]
        all_result.append(classes_result)
        all_percents.append(percent)

    all_result = np.array(all_result)

    for column_id in range(all_result.shape[1]):
        class_result = all_result[:, column_id]

        plt.plot(all_percents, class_result,
                 label=classes[column_id])

    plt.xlabel("Training Size (#samples)")
    plt.ylabel(ylabel)
    plt.legend(loc="best")

    plt.grid(True)
    plt.tight_layout()

    if save_root is not None:
        plt.savefig(save_root)

    plt.close()


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


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_score_bars(ax, all_classes_prob, classes, ignore_label=True):
    classes = list(map(lambda x: x.lower(), classes))

    labels = ['classes']
    x = np.arange(len(labels))
    width = 0.05

    num_classes = len(classes)

    for idx, prob in enumerate(all_classes_prob.tolist()):
        class_score = [prob]
        if ignore_label:
            rect = ax.bar(idx*width + width/2, class_score, width)
        else:
            rect = ax.bar(idx*width + width/2, class_score, width, label=classes[idx])

        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([])

    # ax.legend(prop={'size': 5})


@torch.no_grad()
def images_to_probs(net, images, multilabel_mode, input_vectors=None):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''

    if input_vectors is None:
        output = net(images)
    else:
        output = net(images, input_vectors, training=False)
    
    # convert output probabilities to predicted class
    if not multilabel_mode:
        _, preds_tensor = torch.max(output, 1)
        preds_tensor = preds_tensor.cpu().numpy()
    else:
        preds_tensor = (torch.sigmoid(output).cpu().numpy() > 0.5).astype(int)
    
    preds = np.squeeze(preds_tensor)


    if isinstance(preds.tolist(), int):
        preds = [preds]

    # probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    if not multilabel_mode:
        all_classes_probs = [F.softmax(el, dim=0).cpu().detach().numpy() for el in output]
    else:
        all_classes_probs = [F.sigmoid(el).cpu().detach().numpy() for el in output]

    return preds, all_classes_probs


def plot_classes_preds(net, images, labels, num_images,
                       multilabel_mode, dataset, input_vectors=None):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    if not multilabel_mode:
        classes = dataset.classes
    else:
        classes = np.concatenate((dataset.classes, dataset.combined_classes))

    classes = list(map(lambda x: x.lower(), classes))

    if input_vectors is None:
        preds, all_classes_probs = images_to_probs(net, images, multilabel_mode)
    else:
        preds, all_classes_probs = images_to_probs(net, images, multilabel_mode, input_vectors)

    width_ratios = [el for _ in range(6) for el in [2, 1]]
    fig, a = plt.subplots(6, 12, figsize=(14, 8),
                        gridspec_kw={'width_ratios': width_ratios,
                                        'height_ratios': [1 for _ in range(6)]})
    fig.tight_layout()

    for k in range(num_images):
        img = images[k].mean(dim=0) # for one-chanel image

        img = img / 2 + 0.5
        img = img.cpu().numpy()

        r = (2*k)//12
        c = (2*k)%12
        a[r, c].axis('off')

        a[r, c].imshow(img, cmap='Greys') # for one-chanel image
        # a[r, c].imshow(np.transpose(npimg, (1, 2, 0))) # for RGB image

        label = labels[k].item()

        if not multilabel_mode:
            pred_class = classes[preds[k]]
        else:
            pred_class = dataset.convert_multilabel_to_label(preds[k])


        a[r, c].set_title("{0}\n(label: {1})".format(
            pred_class,
            classes[label]),
                          color=("green" if pred_class==classes[label] else "red"),
                          fontsize=6)

        show_score_bars(a[(2*k+1)//12, (2*k+1)%12],
                        all_classes_probs[k],
                        classes=classes,
                        ignore_label=(k!=0))


    fig.legend(loc="upper center", ncol=7, prop={'size': 6})

    return fig

@torch.no_grad()
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


if __name__ == '__main__':
    MASS_SHAPE_TRAINING_SIZE = 875
    MASS_MARGINS_TRAINING_SIZE = 817
    MASS_BREAST_DENSITY_LESION_TRAINING_SIZE = 932
    MASS_BREAST_DENSITY_IMAGE_TRAINING_SIZE = 932

    CALC_TYPE_TRAINING_SIZE = 868
    CALC_DIST_TRAINING_SIZE = 805
    CALC_BREAST_DENSITY_LESION_TRAINING_SIZE = 981
    CALC_BREAST_DENSITY_IMAGE_TRAINING_SIZE = 981
    

    # Mass Shape
    mass_shape_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
    mass_shape_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 17:00:47 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Sat Sep 25 03:26:16 CDT 2021',
    ]
    mass_shape_dirs = [os.path.join(mass_shape_result_root, resdir)
                         for resdir in mass_shape_dirs]
    mass_shape_dirs = [(resdir, (10-idx)*0.1*MASS_SHAPE_TRAINING_SIZE) for idx, resdir
                       in enumerate(mass_shape_dirs)]
    mass_shape_summary_root = os.path.join(mass_shape_result_root, 'summary')
    os.makedirs(mass_shape_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(mass_shape_result_root, 'summary/mass_shape_roc_aucs.png'),
                        'classes_aucs_list', 'mass_shape',
                        *mass_shape_dirs)

    # Mass Margins
    mass_margins_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
    mass_margins_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 17:29:06 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Sun Sep 26 04:01:56 CDT 2021',
    ]
    mass_margins_dirs = [os.path.join(mass_margins_result_root, resdir)
                         for resdir in mass_margins_dirs]
    mass_margins_dirs = [(resdir, (10-idx)*0.1*MASS_MARGINS_TRAINING_SIZE) for idx, resdir in enumerate(mass_margins_dirs)]
    mass_margins_summary_root = os.path.join(mass_margins_result_root, 'summary')
    os.makedirs(mass_margins_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(mass_margins_summary_root, 'mass_margins_roc_aucs.png'),
                        'classes_aucs_list', 'mass_margins',
                        *mass_margins_dirs)


    # Mass Breast Density Lesion
    mass_breast_density_lesion_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion'
    mass_breast_density_lesion_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 17:58:05 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Sun Sep 26 07:04:51 CDT 2021',
    ]
    mass_breast_density_lesion_dirs = [os.path.join(mass_breast_density_lesion_result_root, resdir) for resdir in mass_breast_density_lesion_dirs]
    mass_breast_density_lesion_dirs = [(resdir, (10-idx)*0.1*MASS_BREAST_DENSITY_LESION_TRAINING_SIZE) for idx, resdir in enumerate(mass_breast_density_lesion_dirs)]
    mass_breast_density_lesion_summary_root = os.path.join(mass_breast_density_lesion_result_root, 'summary')
    os.makedirs(mass_breast_density_lesion_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(mass_breast_density_lesion_summary_root, 'mass_breast_density_lesion_roc_aucs.png'),
                        'classes_aucs_list', 'breast_density',
                        *mass_breast_density_lesion_dirs)

    # Mass Breast Density Image
    mass_breast_density_image_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_image'
    mass_breast_density_image_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 09:37:33 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Mon Sep 27 00:52:15 CDT 2021',
    ]
    mass_breast_density_image_dirs = [os.path.join(mass_breast_density_image_result_root, resdir) for resdir in mass_breast_density_image_dirs]
    mass_breast_density_image_dirs = [(resdir, (10-idx)*0.1*MASS_BREAST_DENSITY_IMAGE_TRAINING_SIZE) for idx, resdir in enumerate(mass_breast_density_image_dirs)]
    mass_breast_density_image_summary_root = os.path.join(mass_breast_density_image_result_root, 'summary')
    os.makedirs(mass_breast_density_image_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(mass_breast_density_image_summary_root, 'mass_breast_density_image_roc_aucs.png'),
                        'classes_aucs_list', 'breast_density',
                        *mass_breast_density_image_dirs)

    # Calc Type 
    calc_type_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
    calc_type_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 18:26:07 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Mon Sep 27 07:56:04 CDT 2021',
    ]
    calc_type_dirs = [os.path.join(calc_type_result_root, resdir) for resdir in calc_type_dirs]
    calc_type_dirs = [(resdir, (10-idx)*0.1*CALC_TYPE_TRAINING_SIZE) for idx, resdir in enumerate(calc_type_dirs)]
    calc_type_summary_root = os.path.join(calc_type_result_root, 'summary')
    os.makedirs(calc_type_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(calc_type_summary_root, 'calc_type_roc_aucs.png'),
                        'classes_aucs_list', 'calc_type',
                        *calc_type_dirs)

    # Calc Dist
    calc_dist_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'
    calc_dist_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 19:07:11 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Mon Sep 27 12:00:04 CDT 2021',
    ]
    calc_dist_dirs = [os.path.join(calc_dist_result_root, resdir) for resdir in calc_dist_dirs]
    calc_dist_dirs = [(resdir, (10-idx)*0.1*CALC_DIST_TRAINING_SIZE) for idx, resdir in enumerate(calc_dist_dirs)]
    calc_dist_summary_root = os.path.join(calc_dist_result_root, 'summary')
    os.makedirs(calc_dist_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(calc_dist_summary_root, 'calc_dist_roc_aucs.png'),
                        'classes_aucs_list', 'calc_dist',
                        *calc_dist_dirs)

    # Calc Breast Density Lesion
    calc_breast_density_lesion_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion'
    calc_breast_density_lesion_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 19:55:02 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Mon Sep 27 16:30:47 CDT 2021',
    ]
    calc_breast_density_lesion_dirs = [os.path.join(calc_breast_density_lesion_result_root, resdir) for resdir in calc_breast_density_lesion_dirs]
    calc_breast_density_lesion_dirs = [(resdir, (10-idx)*0.1*CALC_BREAST_DENSITY_LESION_TRAINING_SIZE) for idx, resdir in enumerate(calc_breast_density_lesion_dirs)]
    calc_breast_density_lesion_summary_root = os.path.join(calc_breast_density_lesion_result_root, 'summary')
    os.makedirs(calc_breast_density_lesion_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(calc_breast_density_lesion_summary_root, 'calc_breast_density_lesion_roc_aucs.png'),
                        'classes_aucs_list', 'breast_density',
                        *calc_breast_density_lesion_dirs)


    # Calc Breast Density Image
    calc_breast_density_image_result_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_image'
    calc_breast_density_image_dirs = [
        'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 12:15:57 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.9_Tue Sep 28 00:47:02 CDT 2021',
    ]
    calc_breast_density_image_dirs = [os.path.join(calc_breast_density_image_result_root, resdir) for resdir in calc_breast_density_image_dirs]
    calc_breast_density_image_dirs = [(resdir, (10-idx)*0.1*CALC_BREAST_DENSITY_IMAGE_TRAINING_SIZE) for idx, resdir in enumerate(calc_breast_density_image_dirs)]
    calc_breast_density_image_summary_root = os.path.join(calc_breast_density_image_result_root, 'summary')
    os.makedirs(calc_breast_density_image_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(calc_breast_density_image_summary_root, 'calc_breast_density_image_roc_aucs.png'),
                        'classes_aucs_list', 'breast_density',
                        *calc_breast_density_image_dirs)
