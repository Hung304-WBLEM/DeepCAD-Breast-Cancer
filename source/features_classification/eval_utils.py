import pickle
import os 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return cm


def evalplot_confusion_matrix(y_true, y_pred, classes, save_root):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, classes)
    plt.savefig(os.path.join(save_root, 'confusion_matrix.png'))
    plt.close()

    # Normalized Confusion Matrix
    plt.figure(figsize=(10, 10))
    norm_cm = plot_confusion_matrix(cm, classes, normalize=True)
    plt.savefig(os.path.join(save_root, 'norm_confusion_matrix.png'))
    plt.close()

    return cm, norm_cm


def plot_precision_recall_curve(binarized_y_true, y_proba_pred, class_name):
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
             label=f"{class_name} (AP={ap})")

    return precisions, recalls, ap


def evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes, save_root):
    precisions_list = []
    recalls_list = []
    average_precisions_list = []

    plt.figure(figsize=(10, 10))
    if binarized_y_true.shape[1] == 1:  # For binary clasification
        precisions, recalls, average_precisions = plot_precision_recall_curve(
            binarized_y_true, y_proba_pred[:, 1], classes[1])  # Assume class 1 is positive
        precisions_list.append(precisions)
        recalls_list.append(recalls)
        average_precisions_list.append(average_precisions)
    else:  # For multiclass classification
        for class_id, class_name in enumerate(classes):
            if np.sum(binarized_y_true[:, class_id]) > 0:
                precisions, recalls, average_precisions = plot_precision_recall_curve(
                    binarized_y_true[:, class_id], y_proba_pred[:, class_id], class_name)
                precisions_list.append(precisions)
                recalls_list.append(recalls)
                average_precisions_list.append(average_precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_root, 'pr_curve.png'))
    plt.close()

    return precisions_list, recalls_list, average_precisions_list


def plot_roc_curve(binarized_y_true, y_proba_pred, class_name):
    fpr, tpr, thresholds = roc_curve(binarized_y_true, y_proba_pred)
    auc = roc_auc_score(binarized_y_true, y_proba_pred)
    auc = round(auc, 2)

    plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc})")

    return fpr, tpr, auc


def evalplot_roc_curve(binarized_y_true, y_proba_pred, classes, save_root):
    fprs_list, tprs_list, aucs_list = [], [], []

    plt.figure(figsize=(10, 10))
    if binarized_y_true.shape[1] == 1:  # For binary classification
        fpr, tpr, auc = plot_roc_curve(
            binarized_y_true, y_proba_pred[:, 1], class_name=classes[1])
        fprs_list.append(fpr)
        tprs_list.append(tpr)
        aucs_list.append(aucs_list)
    else:  # For multiclass classification
        for class_id, class_name in enumerate(classes):
            if np.sum(binarized_y_true[:, class_id]) > 0:
                fpr, tpr, auc = plot_roc_curve(
                    binarized_y_true[:, class_id], y_proba_pred[:, class_id], class_name)
                fprs_list.append(fpr)
                tprs_list.append(tpr)
                aucs_list.append(aucs_list)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="best")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_root, 'roc_curve.png'))
    plt.close()

    return fprs_list, tprs_list, aucs_list


def eval_all(y_true, y_proba_pred, classes, save_root):
    matplotlib.use('Agg')

    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])
    y_pred = y_proba_pred.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)

    cm, norm_cm = evalplot_confusion_matrix(y_true, y_pred, classes, save_root)
    precisions_list, recalls_list, average_precisions_list = evalplot_precision_recall_curve(
        binarized_y_true, y_proba_pred, classes, save_root)
    fprs_list, tprs_list, aucs_list = evalplot_roc_curve(
        binarized_y_true, y_proba_pred, classes, save_root)

    plot_data = {'accuracy': acc,
                 'confusion_matrix': cm, 'normalized_confusion_matrix': norm_cm,
                 'classes_precisions_list': precisions_list,
                 'classes_recalls_list': recalls_list,
                 'classes_average_precisions_list': average_precisions_list,
                 'classes_fprs_list': fprs_list,
                 'classes_tprs_list': tprs_list,
                 'classes_aucs_list': aucs_list}

    with open(os.path.join(save_root, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)
