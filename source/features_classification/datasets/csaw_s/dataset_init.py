import os
import torch

from config.cfg_loader import proj_paths_json
# from features_classification.config_origin import options
from features_classification.augmentation.augmentation_funcs import torch_aug, albumentations_aug, augmix_aug


def initialize(options, data_transforms):
    if options.dataset in ['csaws']:
        from features_classification.datasets.csaw_s.csaws_datasets import CSAWS_Dataset as data

    # Get classes
    classes = data.classes

    # Load Dataset root
    data_root = proj_paths_json['DATA']['root']
    csaws_root = os.path.join(
        data_root, proj_paths_json['DATA']['CSAW-S']['root'])

    if options.dataset in ['csaws']:
        cancer_data_dir = os.path.join(csaws_root, proj_paths_json['DATA']['CSAW-S']['cancer'])
        calc_data_dir = os.path.join(csaws_root, proj_paths_json['DATA']['CSAW-S']['calcifications'])
        axillary_data_dir = os.path.join(csaws_root, proj_paths_json['DATA']['CSAW-S']['axillary_lymph_nodes'])
        bg_data_dir = os.path.join(csaws_root, proj_paths_json['DATA']['CSAW-S']['background']['bg_tfds'])

    # Create dataset
    if options.dataset in ['csaws']:
        image_datasets = {'train': data(cancer_data_dir,
                                        calc_data_dir,
                                        axillary_data_dir,
                                        bg_data_dir,
                                        transform=data_transforms['train']
                                        )
                          }

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
