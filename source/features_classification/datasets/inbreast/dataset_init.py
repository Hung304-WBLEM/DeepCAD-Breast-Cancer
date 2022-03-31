import os
import torch

from config.cfg_loader import proj_paths_json
# from features_classification.config_origin import options
from features_classification.augmentation.augmentation_funcs import torch_aug, albumentations_aug, augmix_aug

def initialize(options, data_transforms):
    if options.dataset in ['inbreast_pathology']:
        from features_classification.datasets.inbreast.inbreast_datasets import INBreast_Pathology_Dataset as data
    elif options.dataset in ['inbreast_pathology_wo_bg']:
        from features_classification.datasets.inbreast.inbreast_datasets import INBreast_Pathology_Without_Background_Dataset as data

    # Get classes
    classes = data.classes

    # Load Dataset root
    data_root = proj_paths_json['DATA']['root']
    inbreast_root = os.path.join(
        data_root, proj_paths_json['DATA']['INbreast']['root'])

    if options.dataset in ['inbreast_pathology', 'inbreast_pathology_wo_bg']:
       mass_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['mass_feats']['mass_pathology']) 
       calc_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['calc_feats']['calc_pathology'])
       cluster_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['cluster_feats']['cluster_pathology'])
       distortion_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['distortion_feats']['distortion_pathology'])
       spiculated_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['spiculated_feats']['spiculated_pathology'])
       asymetry_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['asymetry_feats']['asymetry_pathology'])
       bg_data_dir = os.path.join(inbreast_root, proj_paths_json['DATA']['INbreast']['background']['bg_tfds'])

    # Create dataset
    if options.dataset in ['inbreast_pathology']:
        image_datasets = {'train': data(mass_data_dir,
                                        calc_data_dir,
                                        spiculated_data_dir,
                                        asymetry_data_dir,
                                        distortion_data_dir,
                                        cluster_data_dir,
                                        bg_data_dir,
                                        transform=data_transforms['train']
                                        )
                          }
    elif options.dataset in ['inbreast_pathology_wo_bg']:
        image_datasets = {'train': data(mass_data_dir,
                                        calc_data_dir,
                                        spiculated_data_dir,
                                        asymetry_data_dir,
                                        distortion_data_dir,
                                        cluster_data_dir,
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
