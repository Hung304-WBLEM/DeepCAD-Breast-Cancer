import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import argparse
import matplotlib
import logging
import glob
import pickle
import random
import math
import albumentations
import importlib
matplotlib.use('Agg')
import mlflow

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from efficientnet_pytorch import EfficientNet

from features_classification.augmentation.augmentation_funcs import torch_aug, albumentations_aug, augmix_aug
from features_classification.augmentation import custom_transforms
from features_classification.datasets import cbis_ddsm
from features_classification.eval.eval_utils import eval_all, evalplot_precision_recall_curve, evalplot_roc_curve, evalplot_confusion_matrix, plot_train_val_loss
from features_classification.eval.eval_utils import plot_classes_preds, add_pr_curve_tensorboard
from features_classification.eval.eval_funcs import final_evaluate
from features_classification.models.model_initializer import initialize_model, set_parameter_requires_grad
from features_classification.train.train_funcs import train_stage
from features_classification.train.train_utils import set_seed
from features_classification.eval.eval_utils import images_to_probs
from features_classification.test.test_funcs import get_all_preds

from config_origin import options
from utilities.fileio import json
from config.cfg_loader import proj_paths_json
from augmentation import augmix_transform
t2t_vit = importlib.import_module('T2T-ViT.models.t2t_vit')
t2t_vit_utils = importlib.import_module('T2T-ViT.utils')


if __name__ == '__main__':
    

    experiment_root = '/home/hqvo2/Projects/Breast_Cancer/experiments/classification/cbis_ddsm'
    mlflow.set_tracking_uri("file://" + experiment_root)
    mlflow.set_experiment(options.experiment_name)
    experiment = mlflow.get_experiment_by_name(options.experiment_name)


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(vars(options))

    save_path = os.path.join(experiment_root, experiment.experiment_id, run_id)
    
    # Data augmentation and normalization for training
    input_size = options.input_size
    if options.augmentation_type == 'torch':
        data_transforms = torch_aug(input_size)
    elif options.augmentation_type == 'albumentations':
        data_transforms = albumentations_aug(input_size)
    elif options.augmentation_type == 'augmix':
        data_transforms = augmix_aug(input_size)
        

    # Create Training, Validation and Test datasets
    dataset, image_datasets, classes = cbis_ddsm.initialize(options, data_transforms)
    classes_weights = image_datasets['train'].get_classes_weights()

    # Fix random seed
    set_seed()

    # save_path = options.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    # TensorBoard Summary Writer
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard_logs'))

    # Models to choose from [resnet, resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = options.model_name

    # Number of classes in the dataset
    num_classes = len(classes.tolist())

    # Batch size for training (change depending on how much memory you have)
    batch_size = options.batch_size

    # Number of epochs to train
    num_epochs = options.epochs

    # Initialize the model for this run
    model = initialize_model(
        options, model_name, num_classes, use_pretrained=options.use_pretrained, ckpt_path=options.ckpt_path)

    print("Initializing Datasets and Dataloaders...")


    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=True, num_workers=options.num_workers,
            drop_last=True # this could cause decrease in performance
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=False, num_workers=options.num_workers),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size,
            shuffle=False,
            worker_init_fn=np.random.seed(42), num_workers=options.num_workers)
    }

    with torch.no_grad():
        samples = next(iter(dataloaders_dict['train']))

        if not options.use_clinical_feats:
            writer.add_graph(model, samples['image'])
        elif options.use_clinical_feats:
            writer.add_graph(model, (samples['image'], samples['feature_vector']))

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)



    # Setup the loss fn
    if options.weighted_classes:
        print('Optimization with classes weighting')

        print('Classes weights:', list(zip(classes, classes_weights)))
        
        classes_weights = torch.from_numpy(
            classes_weights).type(torch.FloatTensor)
        if options.criterion == 'ce':
            criterion = nn.CrossEntropyLoss(weight=classes_weights.to(device))
        elif options.criterion == 'bce':
            criterion = nn.BCEWithLogitsLoss(pos_weight=classes_weights.to(device))
    else:
        print('Optimization without classes weighting')
        if options.criterion == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif options.criterion == 'bce':
            criterion = nn.BCEWithLogitsLoss()

    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    # 1st stage
    if options.one_stage_training:
        model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_stage(options, model, model_name, criterion,
                        optimizer_type=options.optimizer,
                        last_frozen_layer=options.first_stage_last_frozen_layer,
                        learning_rate=options.first_stage_learning_rate,
                        weight_decay=options.first_stage_weight_decay,
                        dataset=dataset,
                        num_epochs=options.epochs,
                        dataloaders_dict=dataloaders_dict,
                        weighted_samples=options.weighted_samples,
                        writer=writer,
                        device=device,
                        classes=classes)
        all_train_losses.extend(train_loss_hist)
        all_val_losses.extend(val_loss_hist)
        all_train_accs.extend(train_acc_hist)
        all_val_accs.extend(val_acc_hist)
    else:
        epochs_1st_stage = int(math.ceil(options.epochs * 0.06))
        model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_stage(options, model, model_name, criterion,
                        optimizer_type=options.optimizer,
                        last_frozen_layer=options.first_stage_last_frozen_layer,
                        learning_rate=options.first_stage_learning_rate,
                        weight_decay=options.first_stage_weight_decay,
                        dataset=dataset,
                        num_epochs=epochs_1st_stage,
                        dataloaders_dict=dataloaders_dict,
                        weighted_samples=options.weighted_samples,
                        writer=writer,
                        device=device,
                        classes=classes)
        all_train_losses.extend(train_loss_hist)
        all_val_losses.extend(val_loss_hist)
        all_train_accs.extend(train_acc_hist)
        all_val_accs.extend(val_acc_hist)

        # 2nd stage
        epochs_2nd_stage = int(math.ceil(options.epochs * 0.2))
        model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_stage(options, model, model_name, criterion,
                        optimizer_type=options.optimizer,
                        last_frozen_layer=options.second_stage_last_frozen_layer,
                        learning_rate=options.second_stage_learning_rate,
                        weight_decay=options.second_stage_weight_decay,
                        dataset=dataset,
                        num_epochs=epochs_2nd_stage, dataloaders_dict=dataloaders_dict,
                        weighted_samples=options.weighted_samples,
                        writer=writer,
                        device=device,
                        classes=classes)
        all_train_losses.extend(train_loss_hist)
        all_val_losses.extend(val_loss_hist)
        all_train_accs.extend(train_acc_hist)
        all_val_accs.extend(val_acc_hist)

        # 3rd stage
        if options.train_with_fourth_stage:
            epochs_3rd_stage = int(math.ceil(options.epochs * 0.4))
        else:
            epochs_3rd_stage = options.epochs - (epochs_1st_stage + epochs_2nd_stage)
        model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_stage(options, model, model_name, criterion,
                        optimizer_type=options.optimizer,
                        last_frozen_layer=options.third_stage_last_frozen_layer,
                        learning_rate=options.third_stage_learning_rate,
                        weight_decay=options.third_stage_weight_decay,
                        dataset=dataset,
                        num_epochs=epochs_3rd_stage, dataloaders_dict=dataloaders_dict,
                        weighted_samples=options.weighted_samples,
                        writer=writer,
                        device=device,
                        classes=classes)
        all_train_losses.extend(train_loss_hist)
        all_val_losses.extend(val_loss_hist)
        all_train_accs.extend(train_acc_hist)
        all_val_accs.extend(val_acc_hist)

        # 4th stage
        if options.train_with_fourth_stage:
            epochs_4th_stage = \
                options.epochs - (epochs_1st_stage + epochs_2nd_stage + epochs_3rd_stage)
            model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
                train_stage(options, model, model_name, criterion,
                            optimizer_type=options.optimizer,
                            # freeze_type='third_freeze',
                            last_frozen_layer=options.fourth_stage_last_frozen_layer,
                            # learning_rate=0.00001, weight_decay=0.01,
                            learning_rate=options.fourth_stage_learning_rate,
                            weight_decay=options.fourth_stage_weight_decay,
                            dataset=dataset,
                            num_epochs=epochs_4th_stage, dataloaders_dict=dataloaders_dict,
                            weighted_samples=options.weighted_samples,
                            writer=writer,
                            device=device,
                            classes=classes)
            all_train_losses.extend(train_loss_hist)
            all_val_losses.extend(val_loss_hist)
            all_train_accs.extend(train_acc_hist)
            all_val_accs.extend(val_acc_hist)


    torch.save(model.state_dict(), os.path.join(save_path, 'ckpt.pth'))

    plot_train_val_loss(options.epochs, all_train_losses, all_val_losses,
                        all_train_accs, all_val_accs, save_path)

    ################### Test Model #############################

    model.eval()

    with torch.no_grad():
        preds, labels, _ = get_all_preds(model, dataloaders_dict['test'], device, writer,
                                         multilabel_mode=(options.criterion=='bce'),
                                         dataset=dataset,
                                         use_clinical_feats=options.use_clinical_feats)


    # my roc curve
    final_evaluate(model, classes, dataloaders_dict['test'], device, writer,
                   multilabel_mode=(options.criterion=='bce'),
                   dataset=dataset,
                   use_clinical_feats=options.use_clinical_feats)
    
    # evaluation
    accuracy, macro_ap, micro_ap, classes_aps, \
        macro_roc_auc, micro_roc_auc, classes_aucs = \
        eval_all(labels.cpu().detach().numpy(),
                 preds.cpu().detach().numpy(), classes, save_path,
                 multilabel_mode=(options.criterion=='bce'),
                 dataset=dataset)

    # scripted_model = torch.jit.script(model)

    mlflow.set_experiment(options.experiment_name)

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.pytorch.log_model(model, "model")

        # model_path = mlflow.get_artifact_uri("model")
        # loaded_pytorch_model = mlflow.pytorch.load_model(model_path)

        mlflow.log_metrics({
            'acc': accuracy,
            'macro_ap': macro_ap,
            'micro_ap': micro_ap,
            'macro_auc': macro_roc_auc,
            'micro_auc': micro_roc_auc
        })

        for class_name, ap in zip(classes, classes_aps):
            mlflow.log_metric('z_ap_' + class_name, ap)

        for class_name, auc in zip(classes, classes_aucs):
            mlflow.log_metric('z_auc_' + class_name, auc)
