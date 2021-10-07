import pickle
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from datasets import Pathology_Dataset, Mass_Calc_Pathology_Dataset, Four_Classes_Mass_Calc_Pathology_Dataset, Five_Classes_Mass_Calc_Pathology_Dataset
from datasets import Four_Classes_Features_Pathology_Dataset, Features_Pathology_Dataset
from datasets import Mass_Shape_Dataset, Mass_Margins_Dataset, Calc_Type_Dataset, Calc_Dist_Dataset, Breast_Density_Dataset


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    norm_cm = np.nan_to_num(norm_cm)

    print(cm)
    print(norm_cm)
    plt.imshow(norm_cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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


def evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes, save_root=None, fig_only=False):
    precisions_list = []
    recalls_list = []
    average_precisions_list = []

    if fig_only:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = plt.figure(figsize=(10, 10))

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

    if save_root is not None:
        plt.savefig(os.path.join(save_root, 'pr_curve.png'))

    if fig_only:
        return fig

    plt.close()
    return precisions_list, recalls_list, average_precisions_list


def plot_roc_curve(binarized_y_true, y_proba_pred, class_name):
    fpr, tpr, thresholds = roc_curve(binarized_y_true, y_proba_pred)
    auc = roc_auc_score(binarized_y_true, y_proba_pred)
    auc = round(auc, 2)

    plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc})")

    return fpr, tpr, auc


def evalplot_roc_curve(binarized_y_true, y_proba_pred, classes, save_root=None, fig_only=False):
    fprs_list, tprs_list, aucs_list = [], [], []

    if fig_only:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = plt.figure(figsize=(10, 10))
    if binarized_y_true.shape[1] == 1:  # For binary classification
        fpr, tpr, auc = plot_roc_curve(
            binarized_y_true, y_proba_pred[:, 1], class_name=classes[1])
        fprs_list.append(fpr)
        tprs_list.append(tpr)
        aucs_list.append(auc)
    else:  # For multiclass classification
        for class_id, class_name in enumerate(classes):
            if np.sum(binarized_y_true[:, class_id]) > 0:
                fpr, tpr, auc = plot_roc_curve(
                    binarized_y_true[:, class_id], y_proba_pred[:, class_id], class_name)
                fprs_list.append(fpr)
                tprs_list.append(tpr)
                aucs_list.append(auc)

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
    return fprs_list, tprs_list, aucs_list


def eval_all(y_true, y_proba_pred, classes, save_root):
    matplotlib.use('Agg')

    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])
    y_pred = y_proba_pred.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)

    cm = evalplot_confusion_matrix(y_true, y_pred, classes, save_root)
    precisions_list, recalls_list, average_precisions_list = evalplot_precision_recall_curve(
        binarized_y_true, y_proba_pred, classes, save_root)
    fprs_list, tprs_list, aucs_list = evalplot_roc_curve(
        binarized_y_true, y_proba_pred, classes, save_root)

    plot_data = {'accuracy': acc,
                 'confusion_matrix': cm,
                 'classes_precisions_list': precisions_list,
                 'classes_recalls_list': recalls_list,
                 'classes_average_precisions_list': average_precisions_list,
                 'classes_fprs_list': fprs_list,
                 'classes_tprs_list': tprs_list,
                 'classes_aucs_list': aucs_list}

    with open(os.path.join(save_root, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(plot_data, f)


def plot_learning_curve(save_root, metric, dataset, *result_paths):
    '''Plot the learning curve for different experimental results directories
    
    Args:
    metric(str) - metric that you want to plot
    dataset(str) - dataset that you want to plot, this is to get information
    about classes of this dataset
    *result_paths - list of tuples, each tuple contains the path and the corresponding data percentage 
    '''

    if dataset == 'mass_shape':
        classes = Mass_Shape_Dataset.classes
    elif dataset == 'mass_margins':
        classes = Mass_Margins_Dataset.classes
    elif dataset == 'calc_type':
        classes = Calc_Type_Dataset.classes
    elif dataset == 'calc_dist':
        classes = Calc_Dist_Dataset.classes
    elif dataset == 'breast_density':
        classes = Breast_Density_Dataset.classes
    elif dataset == 'pathology':
        classes = Pathology_Dataset.classes
    elif dataset == 'mass_calc_pathology':
        classes = Mass_Calc_Pathology_Dataset.classes
    elif dataset == 'four_classes_mass_calc_pathology':
        classes = Four_Classes_Mass_Calc_Pathology_Dataset.classes
    elif dataset == 'five_classes_mass_calc_pathology':
        classes = Five_Classes_Mass_Calc_Pathology_Dataset.classes
    elif dataset == 'four_classes_features_pathology':
        classes = Four_Classes_Features_Pathology_Dataset.classes
    elif dataset == 'features_pathology_dataset':
        classes = Features_Pathology_Dataset.classes

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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Sat Sep 25 02:58:21 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Sat Sep 25 20:30:15 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Sat Sep 25 02:33:53 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Sat Sep 25 02:08:23 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Sat Sep 25 01:44:12 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Sat Sep 25 01:24:35 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Sat Sep 25 01:04:02 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Sat Sep 25 00:44:00 CDT 2021',
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Sun Sep 26 03:38:33 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Sun Sep 26 03:15:33 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Sun Sep 26 02:55:00 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Sun Sep 26 02:34:10 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Sun Sep 26 02:16:17 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Sun Sep 26 01:57:28 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Sun Sep 26 01:42:25 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Sun Sep 26 01:25:50 CDT 2021',
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Sun Sep 26 06:41:45 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Sun Sep 26 06:19:32 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Sun Sep 26 05:57:59 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Sun Sep 26 05:37:28 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Sun Sep 26 05:18:06 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Sun Sep 26 04:59:56 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Sun Sep 26 04:42:52 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Sun Sep 26 04:27:03 CDT 2021'
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Sun Sep 26 22:26:01 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Sun Sep 26 20:00:58 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Sun Sep 26 17:37:57 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Sun Sep 26 15:28:05 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Sun Sep 26 13:19:19 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Sun Sep 26 11:19:06 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Sun Sep 26 09:29:12 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Sun Sep 26 07:28:40 CDT 2021'
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Mon Sep 27 07:15:32 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Mon Sep 27 06:36:51 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Mon Sep 27 06:03:21 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Mon Sep 27 05:29:27 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Mon Sep 27 04:56:59 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Mon Sep 27 04:26:35 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Mon Sep 27 03:59:21 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Mon Sep 27 03:32:39 CDT 2021',
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Mon Sep 27 11:29:32 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Mon Sep 27 10:58:24 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Mon Sep 27 10:32:14 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Mon Sep 27 10:05:16 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Mon Sep 27 09:42:27 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Mon Sep 27 09:19:01 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Mon Sep 27 08:57:04 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Mon Sep 27 08:36:05 CDT 2021',
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Mon Sep 27 15:50:32 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Mon Sep 27 15:15:26 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Mon Sep 27 14:43:02 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Mon Sep 27 14:12:58 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Mon Sep 27 13:45:35 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Mon Sep 27 13:20:13 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Mon Sep 27 12:57:15 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Mon Sep 27 12:34:29 CDT 2021'
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
        'r50_b32_e100_224x224_adam_wc_ws_tr0.8_Mon Sep 27 22:23:22 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.7_Mon Sep 27 20:05:56 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.6_Mon Sep 27 17:49:51 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.5_Mon Sep 27 15:48:54 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.4_Mon Sep 27 22:41:49 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.3_Mon Sep 27 20:42:58 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.2_Mon Sep 27 18:58:30 CDT 2021',
        'r50_b32_e100_224x224_adam_wc_ws_tr0.1_Mon Sep 27 17:13:13 CDT 2021',
    ]
    calc_breast_density_image_dirs = [os.path.join(calc_breast_density_image_result_root, resdir) for resdir in calc_breast_density_image_dirs]
    calc_breast_density_image_dirs = [(resdir, (10-idx)*0.1*CALC_BREAST_DENSITY_IMAGE_TRAINING_SIZE) for idx, resdir in enumerate(calc_breast_density_image_dirs)]
    calc_breast_density_image_summary_root = os.path.join(calc_breast_density_image_result_root, 'summary')
    os.makedirs(calc_breast_density_image_summary_root, exist_ok=True)
    plot_learning_curve(os.path.join(calc_breast_density_image_summary_root, 'calc_breast_density_image_roc_aucs.png'),
                        'classes_aucs_list', 'breast_density',
                        *calc_breast_density_image_dirs)
