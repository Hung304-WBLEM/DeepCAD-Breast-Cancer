import os
import torch

from config.cfg_loader import proj_paths_json
# from features_classification.config_origin import options
from features_classification.augmentation.augmentation_funcs import torch_aug, albumentations_aug, augmix_aug


def initialize(options, data_transforms):
    if options.dataset in ['BCDR-F01_dataset', 'BCDR-F02_dataset',
                           'BCDR-F03_dataset', 'BCDR-D01_dataset',
                           'BCDR-D02_dataset']:
        from features_classification.datasets.bcdr.bcdr_datasets import BCDR_Pathology_Dataset as data
    elif options.dataset in ['All-BCDR']:
        from features_classification.datasets.bcdr.bcdr_datasets import All_BCDR_Pathology_Dataset as data

    # Get classes
    classes = data.classes

    # Load Dataset root
    data_root = proj_paths_json['DATA']['root']
    bcdr_root = os.path.join(
        data_root, proj_paths_json['DATA']['BCDR']['root'])

    mamm_type = options.dataset.split('-')[1][0]
    if mamm_type == 'F':
        mamm_type = 'film'
    elif mamm_type == 'D':
        mamm_type = 'digital'

    if options.dataset in ['BCDR-F01_dataset', 'BCDR-F02_dataset',
                           'BCDR-F03_dataset', 'BCDR-D01_dataset',
                           'BCDR-D02_dataset']:
        mass_data_dir = os.path.join(bcdr_root,
                                     proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                     options.dataset,
                                     proj_paths_json['DATA']['BCDR'][mamm_type]['mass_pathology'])
        calc_data_dir = os.path.join(bcdr_root,
                                     proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                     options.dataset,
                                     proj_paths_json['DATA']['BCDR'][mamm_type]['calc_pathology'])
        microcalc_data_dir = os.path.join(bcdr_root,
                                          proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                          options.dataset,
                                          proj_paths_json['DATA']['BCDR'][mamm_type]['microcalc_pathology'])
        masscalc_data_dir = os.path.join(bcdr_root,
                                         proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                         options.dataset,
                                         proj_paths_json['DATA']['BCDR'][mamm_type]['mass_calc_pathology'])
        massmicrocalc_data_dir = os.path.join(bcdr_root,
                                              proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                              options.dataset,
                                              proj_paths_json['DATA']['BCDR'][mamm_type]['mass_microcalc_pathology'])
        calcmicrocalc_data_dir = os.path.join(bcdr_root,
                                              proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                              options.dataset,
                                              proj_paths_json['DATA']['BCDR'][mamm_type]['calc_microcalc_pathology'])
        bg_data_dir = os.path.join(bcdr_root,
                                   proj_paths_json['DATA']['BCDR'][mamm_type]['root'],
                                   options.dataset,
                                   proj_paths_json['DATA']['BCDR'][mamm_type]['background']['bg_tfds'])
    elif options.dataset in ['All-BCDR']:
        film_data_dir = os.path.join(bcdr_root,
                                     proj_paths_json['DATA']['BCDR']['film']['root'])
        digital_data_dir = os.path.join(bcdr_root,
                                        proj_paths_json['DATA']['BCDR']['digital']['root'])
        data_type = 'pathology'
        
    # Create dataset
    if options.dataset in ['BCDR-F01_dataset', 'BCDR-F02_dataset',
                           'BCDR-F03_dataset', 'BCDR-D01_dataset',
                           'BCDR-D02_dataset']:
        image_datasets = {'train': data(mass_data_dir,
                                        calc_data_dir,
                                        microcalc_data_dir,
                                        masscalc_data_dir,
                                        massmicrocalc_data_dir,
                                        calcmicrocalc_data_dir,
                                        bg_data_dir,
                                        transform=data_transforms['train'])}
    elif options.dataset in ['All-BCDR']:
        image_datasets = {'train': data(film_data_dir,
                                        digital_data_dir,
                                        data_type,
                                        transform=data_transforms['train'])}

    return data, image_datasets, classes


if __name__ == '__main__':
    input_size = 224
    data_transforms = torch_aug(input_size)
    data, image_datasets, classes = initialize(options, data_transforms)
    print(classes)

    dataloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=1)

    freq_dict = dict()
    for sample in dataloader:
        freq_dict[classes[sample['label']]] = freq_dict.get(classes[sample['label']], 0) + 1

    for key, value in freq_dict.items():
        print(key, value)
