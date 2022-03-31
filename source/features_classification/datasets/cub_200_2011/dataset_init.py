import os
import torch

from config.cfg_loader import proj_paths_json
# from features_classification.config_origin import options
from features_classification.augmentation.augmentation_funcs import torch_aug, albumentations_aug, augmix_aug

def initialize(options, data_transforms):
    if options.dataset in ['cub_200_2011']:
        from features_classification.datasets.cub_200_2011.cub_200_2011_datasets import CUB_Dataset as data

    # Get classes
    classes = data.classes

    # Load Dataset root
    data_root = proj_paths_json['DATA']['root']
    cub_root = os.path.join(
        data_root, proj_paths_json['DATA']['CUB-200-2011']['root'])

    if options.dataset in ['cub_200_2011']:
      data_dir = cub_root       

    # Create dataset
    if options.dataset in ['cub_200_2011']:
        train_image_datasets = {'train': data(data_dir, transform=data_transforms['train'])}
        val_image_datasets = {'val': data(data_dir, is_train=False, transform=data_transforms['test'])}
        test_image_datasets = {'test': data(data_dir, is_train=False, transform=data_transforms['test'])}
        
        image_datasets = {**train_image_datasets, **val_image_datasets, **test_image_datasets}

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