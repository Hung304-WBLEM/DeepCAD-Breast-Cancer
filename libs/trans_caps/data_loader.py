import numpy as np
import os

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from norb import smallNORBViewPoint, smallNORB
from config.cfg_loader import proj_paths_json
from features_classification import custom_transforms

def cbis_ddsm_get_dataloaders(dataset, batch_size):
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])

    # Import dataset
    if dataset in ['mass_pathology', 'calc_pathology']:
        from features_classification.datasets import Pathology_Dataset as data
    elif dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
        from features_classification.datasets import Mass_Calc_Pathology_Dataset as data
    elif dataset == 'mass_shape_comb_feats_omit':
        from features_classification.datasets import Mass_Shape_Dataset as data
    elif dataset == 'mass_margins_comb_feats_omit':
        from features_classification.datasets import Mass_Margins_Dataset as data
    elif dataset == 'calc_type_comb_feats_omit':
        from features_classification.datasets import Calc_Type_Dataset as data
    elif dataset == 'calc_dist_comb_feats_omit':
        from features_classification.datasets import Calc_Dist_Dataset as data
    elif dataset in ['mass_breast_density_lesion', 'mass_breast_density_image', 'calc_breast_density_lesion', 'calc_breast_density_image']:
        from features_classification.datasets import Breast_Density_Dataset as data
    elif dataset in ['four_classes_mass_calc_pathology']:
        from features_classification.datasets import Four_Classes_Mass_Calc_Pathology_Dataset as data

    # Get classes
    classes = data.classes


    if dataset in ['mass_pathology', 'mass_pathology_clean', 'mass_shape_comb_feats_omit', 'mass_margins_comb_feats_omit', 'mass_breast_density_lesion', 'mass_breast_density_image']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][args.dataset])

    elif dataset in ['calc_pathology', 'calc_pathology_clean', 'calc_type_comb_feats_omit', 'calc_dist_comb_feats_omit', 'calc_breast_density_lesion', 'calc_breast_density_image']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][args.dataset])

    elif dataset in ['mass_calc_pathology', 'four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])

    elif dataset in ['stoa_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['stoa_mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['stoa_calc_pathology'])

    # Data augmentation and normalization for training
    # Just normalization for validation
    # input_size = 512 # remove after experiment
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(25, scale=(0.8, 1.2)),
            custom_transforms.IntensityShift((-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Get datasets
    # Create Training, Validation and Test datasets
    if dataset in ['mass_calc_pathology', 'four_classes_mass_calc_pathology', 'stoa_mass_calc_pathology']:
        image_datasets = {x: data(os.path.join(mass_data_dir, x),
                                  os.path.join(calc_data_dir, x),
                                  transform=data_transforms[x])
                          for x in ['train', 'val', 'test']}
    else:
        image_datasets = {x: data(os.path.join(data_dir, x),
                                  data_transforms[x])
                          for x in ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, worker_init_fn=np.random.seed(42), shuffle=True, num_workers=0) for x in ['train', 'val']}
    
    test_dataloaders_dict = {'test': torch.utils.data.DataLoader(
        image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0)}

    return dataloaders_dict['train'], dataloaders_dict['val'], test_dataloaders_dict['test']
    

def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transforms.Compose(trans))

    elif dataset == "cifar100":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                    transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='train', download=True,
                                transform=transforms.Compose(trans))

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                 transforms.RandomCrop(32),
                 transforms.ColorJitter(brightness=32. / 255, contrast=0.3),
                 transforms.ToTensor(),
                 # transforms.Normalize((0.7199,), (0.117,))
                 ]
        if exp in VIEWPOINT_EXPS:
            train_set = smallNORBViewPoint(data_dir, exp=exp, train=True, download=True,
                                           transform=transforms.Compose(trans))
            trans = trans[:1] + [transforms.CenterCrop(32)] + trans[3:]
            valid_set = smallNORBViewPoint(data_dir, exp=exp, train=False, familiar=False, download=False,
                                           transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=True, download=True, transform=transforms.Compose(trans))

    if exp not in VIEWPOINT_EXPS:
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    exp='azimuth',  # smallnorb only
                    familiar=True,  # smallnorb only
                    num_workers=4,
                    pin_memory=False):
    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                   transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='test', download=True,
                                transform=transforms.Compose(trans))

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                 transforms.CenterCrop(32),
                 transforms.ToTensor(),
                 # transforms.Normalize((0.7199,), (0.117,))
                 ]
        if exp in VIEWPOINT_EXPS:
            dataset = smallNORBViewPoint(data_dir, exp=exp, familiar=familiar, train=False, download=True,
                                         transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=False, download=True,
                                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']
