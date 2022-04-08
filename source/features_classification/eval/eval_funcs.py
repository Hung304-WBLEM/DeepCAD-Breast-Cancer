import torch
import numpy as np

from features_classification.test.test_funcs import get_all_preds
from features_classification.eval.eval_utils import eval_all, evalplot_precision_recall_curve, evalplot_roc_curve, evalplot_confusion_matrix

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score


@torch.no_grad()
def evaluate(model, classes, dataloader, device, writer, epoch,
             multilabel_mode, dataset, eval_split, use_clinical_feats=False,
             use_clinical_feats_only=False):
    model.eval()

    num_classes = len(classes)

    # with torch.no_grad():
    if not (use_clinical_feats or use_clinical_feats_only):
        preds, labels, _ = get_all_preds(model, dataloader, device, writer,
                                            multilabel_mode,
                                            dataset)
    elif use_clinical_feats_only:
        preds, labels, _ = get_all_preds(model, dataloader, device, writer,
                                         multilabel_mode,
                                         dataset, use_clinical_feats=use_clinical_feats,
                                         use_clinical_feats_only=use_clinical_feats_only)
    else:
        preds, labels, _ = get_all_preds(model, dataloader, device, writer,
                                            multilabel_mode,
                                            dataset, use_clinical_feats=use_clinical_feats)

    if not multilabel_mode:
        y_proba_pred = torch.softmax(preds, dim=-1)
    else:
        y_proba_pred_softmax = torch.softmax(preds, dim=-1)
        y_proba_pred_softmax = y_proba_pred_softmax.cpu().detach().numpy()
        y_proba_pred = torch.sigmoid(preds)


    y_true = labels.cpu().detach().numpy()
    y_proba_pred = y_proba_pred.cpu().detach().numpy()
    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])

    if not multilabel_mode:
        y_pred = y_proba_pred.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
    else:
        y_pred = y_proba_pred_softmax.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)

        # ... Need to add something for other metrics like AUC
    _, _, _, pr_log_info = \
        evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes)
    _, _, _, roc_log_info = \
        evalplot_roc_curve(binarized_y_true, y_proba_pred, classes)

    macro_ap = pr_log_info['macro_average_precision']
    micro_ap = pr_log_info['micro_average_precision']
    macro_auc = roc_log_info['macro_roc_auc']
    micro_auc = roc_log_info['micro_roc_auc']

    # accuracy
    writer.add_scalar(f'{eval_split} acc', acc, epoch)
    writer.add_scalar(f'{eval_split} macro ap', macro_ap, epoch)
    writer.add_scalar(f'{eval_split} micro ap', micro_ap, epoch)
    writer.add_scalar(f'{eval_split} macro auc', macro_auc, epoch)
    writer.add_scalar(f'{eval_split} micro auc', micro_auc, epoch)

    # AUCs
    # _, _, pr_aucs = evalplot_precision_recall_curve(binarized_y_true, y_proba_pred, classes)
    # _, _, roc_aucs = evalplot_roc_curve(binarized_y_true, y_proba_pred, classes)

    # idx = 0
    # for class_id, class_name in enumerate(classes):
    #     if np.sum(binarized_y_true[:, class_id]) > 0:
    #         writer.add_scalar(f'test pr auc - {class_name}', pr_aucs[idx], epoch)
    #         writer.add_scalar(f'test roc auc - {class_name}', roc_aucs[idx], epoch)
    #         idx += 1

    model.train()

    return acc, macro_ap, micro_ap, macro_auc, micro_auc


@torch.no_grad()
def final_evaluate(model, classes, test_dataloader, device, writer,
                   multilabel_mode, dataset, use_clinical_feats=False,
                   use_clinical_feats_only=False):
    model.eval()

    num_classes = len(classes)

    with torch.no_grad():
        if not (use_clinical_feats or use_clinical_feats_only):
            preds, labels, _ = get_all_preds(model, test_dataloader, device, writer,
                                                multilabel_mode,
                                                dataset, plot_test_images=True)
        elif use_clinical_feats_only:
            preds, labels, _ = get_all_preds(model, test_dataloader, device, writer,
                                             multilabel_mode,
                                             dataset, plot_test_images=True,
                                             use_clinical_feats_only=use_clinical_feats_only)
        else:
            preds, labels, _ = get_all_preds(model, test_dataloader, device, writer,
                                             multilabel_mode,
                                             dataset, plot_test_images=True,
                                             use_clinical_feats=True)

        if not multilabel_mode:
            y_proba_pred = torch.softmax(preds, dim=-1)
        else:
            y_proba_pred_softmax = torch.softmax(preds, dim=-1).cpu()
            y_proba_pred = torch.sigmoid(preds)
            
        # binarized_labels = label_binarize(
        #     labels.cpu(), classes=[*range(num_classes)])

    y_true = labels.cpu().detach().numpy()
    y_proba_pred = y_proba_pred.cpu().detach().numpy()
    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])

    if not multilabel_mode:
        y_pred = y_proba_pred.argmax(axis=1)
    else:
        y_pred = y_proba_pred_softmax.argmax(axis=1)
        # ... Need to add something for other metrics like AUC

    if hasattr(dataset, 'combined_classes'):
        all_classes = np.concatenate((classes, dataset.combined_classes))
    else:
        all_classes = classes

    writer.add_figure(f'test confusion matrix',
                      evalplot_confusion_matrix(y_true,
                                                y_pred, all_classes, fig_only=True),
                        global_step=None)
    writer.add_figure(f'test roc curve',
                        evalplot_roc_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)
    writer.add_figure(f'test pr curve',
                        evalplot_precision_recall_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)
