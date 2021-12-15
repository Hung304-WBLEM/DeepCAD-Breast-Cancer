import os

from config.cfg_loader import proj_paths_json

def initialize(options, data_transforms):
    if options.dataset in ['mass_pathology', 'calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Pathology_Dataset as data
    elif options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['four_classes_mass_calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Four_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['five_classes_mass_calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Five_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['mass_shape']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Shape_Dataset as data
    elif options.dataset in ['mass_margins']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Margins_Dataset as data
    elif options.dataset in ['calc_type']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Calc_Type_Dataset as data
    elif options.dataset in ['calc_dist']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Calc_Dist_Dataset as data
    elif options.dataset in ['breast_density_image']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Breast_Density_Dataset as data

    # Get classes
    classes = data.classes

    # Load Dataset root
    data_root = proj_paths_json['DATA']['root']
    processed_cbis_ddsm_root = os.path.join(
        data_root, proj_paths_json['DATA']['processed_CBIS_DDSM'])


    if options.dataset in ['mass_pathology',
                           'mass_shape',
                           'mass_margins']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][options.dataset])

    elif options.dataset in ['calc_pathology',
                             'calc_type',
                             'calc_dist']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats'][options.dataset])

    elif options.dataset in ['breast_density_image']:
        
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_breast_density_image'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_breast_density_image'])

    elif options.dataset in ['mass_calc_pathology',
                             'four_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])


    elif options.dataset in ['five_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_tfds'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_tfds'])
        bg_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['background']['bg_tfds'])
        

    elif options.dataset in ['stoa_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['stoa_mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['stoa_calc_pathology'])


    # Create Training, Validation and Test datasets
    if options.dataset in ['breast_density_image',
                           'mass_calc_pathology',
                           'four_classes_mass_calc_pathology',
                           'stoa_mass_calc_pathology']:
        train_image_datasets = {'train': data(os.path.join(mass_data_dir, 'train'),
                                    os.path.join(calc_data_dir, 'train'),
                                    transform=data_transforms['train'],
                                    train_rate=options.train_rate)
                                }
        val_test_image_datasets = {x: data(os.path.join(mass_data_dir, x),
                                        os.path.join(calc_data_dir, x),
                                        transform=data_transforms[x])
                                   for x in ['val', 'test']}
        image_datasets = {**train_image_datasets, **val_test_image_datasets}
    elif options.dataset in ['five_classes_mass_calc_pathology']:
        train_image_datasets = {'train': data(os.path.join(mass_data_dir, 'train'),
                                    os.path.join(calc_data_dir, 'train'),
                                    os.path.join(bg_data_dir, 'train'),
                                    transform=data_transforms['train'])
                                }
        val_test_image_datasets = {x: data(os.path.join(mass_data_dir, x),
                                        os.path.join(calc_data_dir, x),
                                        os.path.join(bg_data_dir, x),
                                        transform=data_transforms[x])
                                   for x in ['val', 'test']}
        image_datasets = {**train_image_datasets, **val_test_image_datasets}
        
    else:
        train_image_datasets = {'train': data(os.path.join(data_dir, 'train'),
                                              data_transforms['train'],
                                              train_rate=options.train_rate)}
        val_test_image_datasets = {x: data(os.path.join(data_dir, x),
                                           data_transforms[x])
                                   for x in ['val', 'test']}
        image_datasets = {**train_image_datasets, **val_test_image_datasets}


    return data, image_datasets, classes 
