import os

from config.cfg_loader import proj_paths_json

def initialize(options, data_transforms):
    if options.dataset in ['mass_pathology', 'calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Pathology_Dataset as data
    elif options.dataset in ['mass_calc_pathology', 'stoa_mass_calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['four_classes_mass_calc_pathology',
                             'four_classes_mass_calc_pathology_birads34',
                             'four_classes_mass_calc_pathology_tfds']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Four_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['four_classes_features_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Four_Classes_Features_Pathology_Dataset as data
    elif options.dataset in ['five_classes_mass_calc_pathology']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Five_Classes_Mass_Calc_Pathology_Dataset as data
    elif options.dataset in ['mass_shape', 'mass_shape_tfds']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Shape_Dataset as data
    elif options.dataset in ['mass_margins', 'mass_margins_tfds']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Mass_Margins_Dataset as data
    elif options.dataset in ['calc_type', 'calc_type_tfds']:
        from features_classification.datasets.cbis_ddsm.cbis_ddsm_datasets import Calc_Type_Dataset as data
    elif options.dataset in ['calc_dist', 'calc_dist_tfds']:
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
                           'mass_margins',
                           'mass_pathology_tfds',
                           'mass_shape_tfds',
                           'mass_margins_tfds']:
        data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats'][options.dataset])

    elif options.dataset in ['calc_pathology',
                             'calc_type',
                             'calc_dist',
                             'calc_pathology_tfds',
                             'calc_type_tfds',
                             'calc_dist_tfds']:
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
    elif options.dataset in ['four_classes_features_pathology']:
        
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])
        mass_annotation_file = \
        {
            'train': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['annotations_file']['train']
            ),
            'val': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['annotations_file']['train']
            ),
            'test': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['annotations_file']['test']
            )
        }
        calc_annotation_file = {
            'train': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['annotations_file']['train']
            ),
            'val': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['annotations_file']['train']
            ),
            'test': os.path.join(
                data_root, processed_cbis_ddsm_root,
                proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['annotations_file']['test']
            )
        }

    elif options.dataset in ['four_classes_mass_calc_pathology_birads34']:
        
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_birads34'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_birads34'])

    elif options.dataset in ['four_classes_mass_calc_pathology_tfds']:
        
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology_tfds'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology_tfds'])

    elif options.dataset in ['five_classes_mass_calc_pathology']:
        mass_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['mass_feats']['mass_pathology'])
        calc_data_dir = os.path.join(
            data_root, processed_cbis_ddsm_root,
            proj_paths_json['DATA']['CBIS_DDSM_lesions']['calc_feats']['calc_pathology'])
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
                           'four_classes_mass_calc_pathology_birads34',
                           'four_classes_mass_calc_pathology_tfds',
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

    elif options.dataset in ['four_classes_features_pathology']:
        train_image_datasets = {'train':
                                data(
                                    mass_annotation_file['train'],
                                    os.path.join(mass_data_dir, 'train'),
                                    calc_annotation_file['train'],
                                    os.path.join(calc_data_dir, 'train'),
                                    transform=data_transforms['train'],
                                    train_rate=options.train_rate
                                )
                                }
        val_test_image_datasets = {x: data(
            mass_annotation_file[x],
            os.path.join(mass_data_dir, x),
            calc_annotation_file[x],
            os.path.join(calc_data_dir, x),
            transform=data_transforms[x]
        ) for x in ['val', 'test']}

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
