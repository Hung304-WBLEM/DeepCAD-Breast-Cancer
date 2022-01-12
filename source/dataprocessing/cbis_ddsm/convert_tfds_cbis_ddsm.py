import os
import glob
import pandas as pd
import shutil
import math


def tfds_cbis_ddsm_to_clinical(train_csv, test_csv,
                               data_path, save_path,
                               clinical_feat, classes):


    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            for img_path in glob.glob(os.path.join(data_path, split, class_name, '*.png')):
                img_name = os.path.basename(img_path)

                img_name, _ = os.path.splitext(img_name)

                dir1, dir2, abnorm, _ = img_name.split('#') 
                abnorm_id = int(abnorm.split('_')[1])


                # Get Matched Index
                match_path = os.path.join(dir1, dir2)

                if split in ['train', 'val']:
                    match_idx = \
                        train_df['image file path'].str.match(r'(^.*' + match_path + '.*)')
                    match_row = \
                        train_df[match_idx & (train_df['abnormality id']==abnorm_id)]
                elif split in ['test']:
                    match_idx = \
                        test_df['image file path'].str.match(r'(^.*' + match_path + '.*)')
                    match_row = \
                        test_df[match_idx & (test_df['abnormality id']==abnorm_id)]
                else:
                    raise ValueError

                clinical_feat_val = match_row[clinical_feat].values[0]
                if pd.isna(clinical_feat_val):
                    continue


                dst_root = os.path.join(save_path, split, clinical_feat_val)
                print(dst_root)

                if not os.path.exists(dst_root):
                    os.makedirs(dst_root, exist_ok=True)

                shutil.copy(img_path, os.path.join(dst_root, img_name+'.png'))

if __name__ == '__main__':
    data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM'

    mass_train_csv = os.path.join(data_root, 'mass_case_description_train_set.csv')
    mass_test_csv = os.path.join(data_root, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(data_root, 'calc_case_description_train_set.csv')
    calc_test_csv = os.path.join(data_root, 'calc_case_description_test_set.csv')

    tfds_data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM_tfds'

    mass_classes = [
        'BENIGN_MASS',
        'MALIGNANT_MASS'
    ]

    # tfds_mass_shape_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM/mass/cls/mass_shape_tfds'
    # tfds_cbis_ddsm_to_clinical(mass_train_csv, mass_test_csv,
    #                            tfds_data_root, tfds_mass_shape_root,
    #                            clinical_feat='mass shape',
    #                            classes=mass_classes)

    # tfds_mass_margins_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM/mass/cls/mass_margins_tfds'
    # tfds_cbis_ddsm_to_clinical(mass_train_csv, mass_test_csv,
    #                            tfds_data_root, tfds_mass_margins_root,
    #                            clinical_feat='mass margins',
    #                            classes=mass_classes)

    calc_classes = [
        'BENIGN_CALCIFICATION',
        'MALIGNANT_CALCIFICATION'
    ]

    # tfds_calc_type_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM/calc/cls/calc_type_tfds'
    # tfds_cbis_ddsm_to_clinical(calc_train_csv, calc_test_csv,
    #                            tfds_data_root, tfds_calc_type_root,
    #                            clinical_feat='calc type',
    #                            classes=calc_classes)

    # tfds_calc_dist_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM/calc/cls/calc_dist_tfds'
    # tfds_cbis_ddsm_to_clinical(calc_train_csv, calc_test_csv,
    #                            tfds_data_root, tfds_calc_dist_root,
    #                            clinical_feat='calc distribution',
    #                            classes=calc_classes)
