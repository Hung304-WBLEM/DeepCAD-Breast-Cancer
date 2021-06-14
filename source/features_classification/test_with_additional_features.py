import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
import time
import copy
import logging
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import custom_transforms
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix
from eval_utils import eval_all
from sklearn.preprocessing import label_binarize
from dataprocessing.process_cbis_ddsm import get_info_lesion
from config.cfg_loader import proj_paths_json
from skimage import io, transform
from torch import nn
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from train_with_additional_features import set_parameter_requires_grad
from datasets import Features_Pathology_Dataset, Four_Classes_Features_Pathology_Dataset
from PIL import Image
from train_utils import compute_classes_weights, compute_classes_weights_mass_calc_pathology, compute_classes_weights_mass_calc_pathology_4class, compute_classes_weights_mass_calc_pathology_5class, set_seed, plot_train_val_loss
from train_with_additional_features import get_all_preds_pathology, train_pathology_model, train_stage_pathology
from train_with_additional_features import Pathology_Model, Attentive_Pathology_Model
from config_fusion import read_config
from config_fusion import options as default_opts


if __name__ == '__main__':
    ##########################################
    ############ Read Options ################
    ##########################################
    options = read_config(config_path=os.path.join(default_opts.save_path, 'config.ini'))
    # options = read_config(config_path='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_b32_e100_224x224_adam_wc_ws_Mon May  3 06:08:38 CDT 2021/config.ini')
    

    ##########################################
    ########## Load Dataset Root #############
    ##########################################
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # Import dataset
    if options.dataset in ['mass_pathology', 'mass_pathology_clean', 'calc_pathology', 'calc_pathology_clean']:
        from datasets import Features_Pathology_Dataset as data
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        from datasets import Four_Classes_Features_Pathology_Dataset as data

    # Get classes
    classes = data.classes

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][options.dataset])
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root, proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][options.dataset])
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

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
    
    # Initialize model
    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5

        if options.fusion_type == 'concat':
            model = Pathology_Model(model_name,
                                    input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats, num_classes=num_classes)
        elif options.fusion_type in ['coatt', 'crossatt']:
            model = \
                Attentive_Pathology_Model(model_name,
                                          input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats,
                                          num_classes=num_classes,
                                          attention_type=options.fusion_type)
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        breast_density_cats = 4
        calc_type_cats = 14
        calc_dist_cats = 5
        if options.fusion_type == 'concat':
            model = Pathology_Model(model_name,
                                    input_vector_dim=breast_density_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
        elif options.fusion_type in ['coatt', 'crossatt']:
            model = Attentive_Pathology_Model(model_name,
                                              input_vector_dim=breast_density_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes, attention_type=options.fusion_type)
            
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        calc_type_cats = 14
        calc_dist_cats = 5

        if options.fusion_type == 'concat':
            model = Pathology_Model(model_name, 
                                    input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes)
        elif options.fusion_type in ['coatt', 'crossatt']:
            model = Attentive_Pathology_Model(model_name, 
                                              input_vector_dim=breast_density_cats+mass_shape_cats+mass_margins_cats+calc_type_cats+calc_dist_cats, num_classes=num_classes, attention_type=options.fusion_type)
        

    # print the model we just instantiated
    print(model)

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

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        pathology_datasets = \
            {x: Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        pathology_datasets = \
            {'train': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'train'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'train'),
                uncertainty=options.train_uncertainty,
                missed_feats_num=options.missed_feats_num,
                transform=data_transforms['train']
            ),
            'val': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/train/mass_case_description_train_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'val'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/train/calc_case_description_train_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'val'),
                uncertainty=options.test_uncertainty,
                missed_feats_num=options.missed_feats_num,
                transform=data_transforms['val']
            )}

    if options.dataset in ['mass_pathology', 'mass_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='mass',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif options.dataset in ['calc_pathology', 'calc_pathology_clean']:
        test_image_datasets = \
            {'test': Features_Pathology_Dataset(lesion_type='calc',
                annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                root_dir=os.path.join(data_dir, 'test'), transform=data_transforms['test'])}
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        test_image_datasets = \
            {'test': Four_Classes_Features_Pathology_Dataset(
                mass_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/mass_case_description_test_set.csv',
                mass_root_dir=os.path.join(mass_data_dir, 'test'),
                calc_annotation_file='/home/hqvo2/Projects/Breast_Cancer/data/processed_data/calc/test/calc_case_description_test_set.csv',
                calc_root_dir=os.path.join(calc_data_dir, 'test'),
                uncertainty=options.test_uncertainty,
                missed_feats_num=options.missed_feats_num,
                transform=data_transforms['test']
            )}
        

    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            pathology_datasets['train'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=True, num_workers=options.num_workers),
        'val': torch.utils.data.DataLoader(
            pathology_datasets['val'], batch_size=batch_size,
            worker_init_fn=np.random.seed(42),
            shuffle=False, num_workers=options.num_workers)
    }

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup the loss fn
    if options.weighted_classes:
        print('Optimization with classes weighting')
        if options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        elif options.dataset in ['four_classes_mass_calc_pathology']:
            classes_weights = compute_classes_weights_mass_calc_pathology_4class(
                mass_root=os.path.join(mass_data_dir, 'train'),
                calc_root=os.path.join(calc_data_dir, 'train'),
                classes_names=classes
            )
        else:
            classes_weights = compute_classes_weights(
                data_root=os.path.join(data_dir, 'train'), classes_names=classes)


        classes_weights = torch.from_numpy(
            classes_weights).type(torch.FloatTensor)
        criterion = nn.CrossEntropyLoss(weight=classes_weights.to(device))
    else:
        print('Optimization without classes weighting')
        criterion = nn.CrossEntropyLoss()


    ################### Test Model #############################
    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(test_image_datasets['test'], batch_size=batch_size, shuffle=False, worker_init_fn=np.random.seed(42), num_workers=options.num_workers)}

    model.load_state_dict(torch.load(os.path.join(save_path, 'ckpt.pth')))
    model.eval()
    
    with torch.no_grad():
        prediction_loader = test_dataloaders_dict['test']
        preds, labels, paths = get_all_preds_pathology(model, prediction_loader, device, use_predicted_feats=default_opts.use_predicted_feats)

        softmaxs = torch.softmax(preds, dim=-1)
        binarized_labels = label_binarize(labels.cpu(), classes=[*range(num_classes)])

    predicted_labels = torch.max(softmaxs, 1).indices.cpu().numpy().tolist()
    labels_list = labels.cpu().numpy().tolist()
    paths = (paths)

    rows = []
    for path, label, predict in zip(paths, labels_list, predicted_labels):
        rows.append([path, str(label), str(predict)])


    df = pd.DataFrame(rows, columns=['Path', 'Label', 'Predict'])

    if default_opts.use_predicted_feats:
        test_root = os.path.join(save_path, 'test_with_predictions')
        os.makedirs(test_root, exist_ok=True)
        df.to_csv(os.path.join(test_root, 'test_preds.csv'))

        # evaluation
        eval_all(labels.cpu().detach().numpy(),
                softmaxs.cpu().detach().numpy(), classes, test_root)
    else:
        test_root = os.path.join(save_path, 'test')
        os.makedirs(test_root, exist_ok=True)
        df.to_csv(os.path.join(test_root, 'test_preds.csv'))

        # evaluation
        eval_all(labels.cpu().detach().numpy(),
                softmaxs.cpu().detach().numpy(), classes, test_root)
