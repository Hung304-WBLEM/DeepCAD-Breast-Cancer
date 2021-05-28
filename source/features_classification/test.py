import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import matplotlib
import logging
import glob
import pickle
import random
import custom_transforms
import pandas as pd
matplotlib.use('Agg')

from eval_utils import eval_all
from utilities.fileio import json
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from torchvision import datasets, models, transforms
from config.cfg_loader import proj_paths_json
from train_utils import compute_classes_weights, compute_classes_weights_mass_calc_pathology, compute_classes_weights_mass_calc_pathology_4class, compute_classes_weights_mass_calc_pathology_5class, set_seed, plot_train_val_loss
from train import get_all_preds, initialize_model
from config_origin import read_config
from config_origin import options as default_opts


if __name__ == '__main__':
    #############################################
    ############# Read Options ##################
    #############################################
    
    options = read_config(config_path=os.path.join(default_opts.save_path, 'config.ini'))
    # options = read_config(config_path='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 15:52:21 CDT 2021/config.ini')

    #############################################
    ############# Load Dataset Root #############
    #############################################
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # Import dataset
    if options.dataset in ['mass_pathology', 'calc_pathology', 'mass_pathology_clean', 'calc_pathology_clean']:
        from datasets import Pathology_Dataset as data
    elif options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
        from datasets import Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['four_classes_mass_calc_pathology', 'four_classes_mass_calc_pathology_512x512-crop_zero-pad', 'four_classes_mass_calc_pathology_1024x1024-crop_zero-pad', 'four_classes_mass_calc_pathology_2048x2048-crop_zero-pad', 'four_classes_mass_calc_pathology_histeq']:
        from datasets import Four_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['five_classes_mass_calc_pathology']:
        from datasets import Five_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['mass_shape_comb_feats_omit',
                             'mass_shape_comb_feats_omit_segm',
                             'mass_shape_comb_feats_omit_mask']:
        from datasets import Mass_Shape_Dataset as data
    elif options.dataset in ['mass_margins_comb_feats_omit',
                             'mass_margins_comb_feats_omit_segm',
                             'mass_margins_comb_feats_omit_mask']:
        from datasets import Mass_Margins_Dataset as data
    elif options.dataset in ['calc_type_comb_feats_omit',
                             'calc_type_comb_feats_omit_segm',
                             'calc_type_comb_feats_omit_mask']:
        from datasets import Calc_Type_Dataset as data
    elif options.dataset in ['calc_dist_comb_feats_omit',
                             'calc_dist_comb_feats_omit_segm',
                             'calc_dist_comb_feats_omit_mask']:
        from datasets import Calc_Dist_Dataset as data
    elif options.dataset in ['mass_breast_density_lesion', 'mass_breast_density_image',
                             'calc_breast_density_lesion', 'calc_breast_density_image',
                             'mass_breast_density_lesion_segm', 'calc_breast_density_lesion_segm',
                             'mass_breast_density_lesion_mask', 'calc_breast_density_lesion_mask']:
        from datasets import Breast_Density_Dataset as data
    
    # Get classes
    classes = data.classes


    if options.dataset in ['mass_pathology', 'mass_pathology_clean',
                           'mass_shape_comb_feats_omit',
                           'mass_margins_comb_feats_omit',
                           'mass_breast_density_lesion',
                           'mass_shape_comb_feats_omit_segm',
                           'mass_margins_comb_feats_omit_segm',
                           'mass_breast_density_lesion_segm',
                           'mass_shape_comb_feats_omit_mask',
                           'mass_margins_comb_feats_omit_mask',
                           'mass_breast_density_lesion_mask',
                           'mass_breast_density_image']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][options.dataset])

    elif options.dataset in ['calc_pathology', 'calc_pathology_clean',
                             'calc_type_comb_feats_omit',
                             'calc_dist_comb_feats_omit',
                             'calc_breast_density_lesion',
                             'calc_type_comb_feats_omit_segm',
                             'calc_dist_comb_feats_omit_segm',
                             'calc_breast_density_lesion_segm',
                             'calc_type_comb_feats_omit_mask',
                             'calc_dist_comb_feats_omit_mask',
                             'calc_breast_density_lesion_mask',
                             'calc_breast_density_image']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][options.dataset])

    elif options.dataset in ['mass_calc_pathology', 'four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

    elif options.dataset in ['four_classes_mass_calc_pathology_histeq']:

        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_histeq'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_histeq'])

    elif options.dataset == 'four_classes_mass_calc_pathology_512x512-crop_zero-pad':
        mass_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_512x512-crop_zero-pad'])

        calc_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_512x512-crop_zero-pad'])

    elif options.dataset == 'four_classes_mass_calc_pathology_1024x1024-crop_zero-pad':
        mass_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_1024x1024-crop_zero-pad'])

        calc_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_1024x1024-crop_zero-pad'])

    elif options.dataset == 'four_classes_mass_calc_pathology_2048x2048-crop_zero-pad':
        mass_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_2048x2048-crop_zero-pad'])

        calc_data_dir = os.path.join(
            data_root,
            processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_2048x2048-crop_zero-pad'])

    elif options.dataset in ['stoa_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['stoa_mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['stoa_calc_pathology'])



    # Fix random seed
    set_seed()

    save_path = options.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO,
                        filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    # TensorBoard Summary Writer
    # writer = SummaryWriter('runs/' + datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    # writer = SummaryWriter(os.path.join(save_path, 'tensorboard_logs'))

    # Models to choose from [resnet, resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = options.model_name

    # Number of classes in the dataset
    num_classes = len(classes.tolist())

    # Batch size for training (change depending on how much memory you have)

    batch_size = options.batch_size

    # Number of epochs to train
    num_epochs = options.epochs

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    # Data augmentation and normalization for training
    input_size = options.input_size
    if options.augmentation_type == 'torch':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(25, scale=(0.8, 1.2)),
                custom_transforms.IntensityShift((-20, 20)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif options.augmentation_type == 'albumentations':
        data_transforms = {
            'train': albumentations.Compose([
                albumentations.Transpose(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
                albumentations.OneOf([
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                albumentations.OneOf([
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                    albumentations.ElasticTransform(alpha=3),
                ], p=0.7),

                albumentations.CLAHE(clip_limit=4.0, p=0.7),
                albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                albumentations.Resize(input_size, input_size),
                albumentations.Cutout(max_h_size=int(input_size * 0.375), max_w_size=int(input_size * 0.375), num_holes=1, p=0.7),
                albumentations.Normalize()
            ]),
            'val': albumentations.Compose([
                albumentations.Resize(input_size, input_size),
                albumentations.Normalize()
            ]),
            'test': albumentations.Compose([
                albumentations.Resize(input_size, input_size),
                albumentations.Normalize()
            ])
        }

    print("Initializing Datasets and Dataloaders...")


    # Create Training, Validation and Test datasets
    if options.dataset in ['mass_calc_pathology', 'four_classes_mass_calc_pathology', 'four_classes_mass_calc_pathology_512x512-crop_zero-pad', 'four_classes_mass_calc_pathology_1024x1024-crop_zero-pad', 'four_classes_mass_calc_pathology_2048x2048-crop_zero-pad', 'four_classes_mass_calc_pathology_histeq', 'stoa_mass_calc_pathology']:
        image_datasets = {x: data(os.path.join(mass_data_dir, x),
                                  os.path.join(calc_data_dir, x),
                                  transform=data_transforms[x])
                          for x in ['train', 'val', 'test']}
    else:
        image_datasets = {x: data(os.path.join(data_dir, x),
                                  data_transforms[x])
                          for x in ['train', 'val', 'test']}


    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    # Setup the loss fn
    if options.weighted_classes:
        print('Optimization with classes weighting')
        if options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        elif options.dataset in ['four_classes_mass_calc_pathology', 'four_classes_mass_calc_pathology_512x512-crop_zero-pad', 'four_classes_mass_calc_pathology_1024x1024-crop_zero-pad', 'four_classes_mass_calc_pathology_2048x2048-crop_zero-pad', 'four_classes_mass_calc_pathology_histeq']:
            classes_weights = compute_classes_weights_mass_calc_pathology_4class(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        else:
            classes_weights = compute_classes_weights(
                data_root=os.path.join(data_dir, 'train'), classes_names=classes)

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

    ################### Test Model #############################
    test_dataloaders_dict = {
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size,
            shuffle=False,
            worker_init_fn=np.random.seed(42), num_workers=0)}

    model_ft.load_state_dict(torch.load(os.path.join(save_path, 'ckpt.pth')))
    model_ft.eval()

    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels, paths = get_all_preds(model_ft, prediction_loader, device)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(
            labels.cpu(), classes=[*range(num_classes)])

    predicted_labels = torch.max(softmaxs, 1).indices.cpu().numpy().tolist()
    labels_list = labels.cpu().numpy().tolist()
    paths = (paths)

    rows = []
    # with open(os.path.join(save_path, 'test_result.txt'), 'w') as f:
    for path, label, predict in zip(paths, labels_list, predicted_labels):
        # f.write(' '.join((path, str(label), str(predict))) + '\n')
        rows.append([path, str(label), str(predict)])

    df = pd.DataFrame(rows, columns=['Path', 'Label', 'Predict'])

    test_root = os.path.join(save_path, 'test')
    os.makedirs(test_root, exist_ok=True)
    df.to_csv(os.path.join(test_root, 'test_preds.csv'))


    # evaluation
    eval_all(labels.cpu().detach().numpy(),
             softmaxs.cpu().detach().numpy(), classes, test_root)
