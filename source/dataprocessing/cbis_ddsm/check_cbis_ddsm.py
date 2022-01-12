import os
import glob
import pandas as pd
import shutil

def check_cbis_ddsm(csv):
    df = pd.read_csv(csv)

    with open(csv + '.log', 'w') as fout:
        for index, row in df.iterrows():
            print(index, end=' ')
            img_path = row['image file path']
            cropped_path = row['cropped image file path']
            roi_path = row['ROI mask file path']


            img_dir = img_path.split('/')[0]
            cropped_dir = cropped_path.split('/')[0]
            roi_dir = cropped_path.split('/')[0]


            if len(glob.glob(os.path.join(data_root, img_dir, '**/**/000000.dcm'))) == 0:
                fout.write(img_dir + '\n')
            if len(glob.glob(os.path.join(data_root, cropped_dir, '**/**/*.dcm'))) < 2:
                fout.write(cropped_dir + '\n')
            if len(glob.glob(os.path.join(data_root, roi_dir, '**/**/*.dcm'))) < 2:
                fout.write(roi_dir + '\n')


def check_cbis_ddsm_tfds(mass_train_csv, mass_test_csv,
                         calc_train_csv, calc_test_csv,
                         data_path, tf_split):

    mass_train_df = pd.read_csv(mass_train_csv)
    mass_test_df = pd.read_csv(mass_test_csv)
    calc_train_df = pd.read_csv(calc_train_csv)
    calc_test_df = pd.read_csv(calc_test_csv)


    classes = [
        'BENIGN_CALCIFICATION', 'BENIGN_MASS',
        'MALIGNANT_CALCIFICATION', 'MALIGNANT_MASS',
        'BACKGROUND'
    ]
    

    for class_name in classes:
        wrong_files = 0
        for img_path in glob.glob(os.path.join(data_path, class_name, '*.png')):
            img_name = os.path.basename(img_path)

            img_name, _ = os.path.splitext(img_name)

            if class_name == 'BACKGROUND':
                dir1, dir2, _ = img_name.split('#') 
            else:
                dir1, dir2, abnorm, _ = img_name.split('#') 
                abnorm_id = int(abnorm.split('_')[1])


            # Get Matched Index
            match_path = os.path.join(dir1, dir2)

            mass_train_match_idx = \
                mass_train_df['image file path'].str.match(r'(^.*' + match_path + '.*)')
            mass_test_match_idx = \
                mass_test_df['image file path'].str.match(r'(^.*' + match_path + '.*)')
            calc_train_match_idx = \
                calc_train_df['image file path'].str.match(r'(^.*' + match_path + '.*)')
            calc_test_match_idx = \
                calc_test_df['image file path'].str.match(r'(^.*' + match_path + '.*)')

            # Get Matched row
            selected_match_row = None

            if 'MASS' in class_name:
                mass_train_match_row = \
                    mass_train_df[mass_train_match_idx & (mass_train_df['abnormality id']==abnorm_id)]
                mass_test_match_row = \
                    mass_test_df[mass_test_match_idx & (mass_test_df['abnormality id']==abnorm_id)]

                if not mass_train_match_row.empty:
                    selected_match_row = mass_train_match_row
                    true_split = 'train'
                elif not mass_test_match_row.empty:
                    selected_match_row = mass_test_match_row
                    true_split = 'test'
                else:
                    raise ValueError
                    
            elif 'CALCIFICATION' in class_name:
                calc_train_match_row = \
                    calc_train_df[calc_train_match_idx & (calc_train_df['abnormality id']==abnorm_id)]
                calc_test_match_row = \
                    calc_test_df[calc_test_match_idx & (calc_test_df['abnormality id']==abnorm_id)]

                if not calc_train_match_row.empty:
                    selected_match_row = calc_train_match_row
                    true_split = 'train'
                elif not calc_test_match_row.empty:
                    selected_match_row = calc_test_match_row
                    true_split = 'test'
                else:
                    raise ValueError

            elif 'BACKGROUND' in class_name:
                bg_mass_train_match_row = \
                    mass_train_df[mass_train_match_idx]
                bg_mass_test_match_row = \
                    mass_test_df[mass_test_match_idx]
                bg_calc_train_match_row = \
                    calc_train_df[calc_train_match_idx]
                bg_calc_test_match_row = \
                    calc_test_df[calc_test_match_idx]

                if not bg_mass_train_match_row.empty:
                    selected_match_row = bg_mass_train_match_row
                    true_split = 'train'

                elif not bg_mass_test_match_row.empty:
                    selected_match_row = bg_mass_test_match_row
                    true_split = 'test'
                    
                elif not bg_calc_train_match_row.empty:
                    selected_match_row = bg_calc_train_match_row
                    true_split = 'train'

                elif not bg_calc_test_match_row.empty:
                    selected_match_row = bg_calc_test_match_row
                    true_split = 'test'
                else:
                    raise ValueError
            else:
                raise ValueError


            save_path = os.path.dirname(data_path)
            if tf_split == 'val':
                if true_split == 'test':
                    src_path = os.path.join(save_path, tf_split, class_name, img_name+'.png')
                    dst_path = os.path.join(save_path, true_split, class_name, img_name+'.png')

                    wrong_files += 1

                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)

            elif tf_split == 'train':
                if true_split == 'test':
                    src_path = os.path.join(save_path, tf_split, class_name, img_name+'.png')
                    dst_path = os.path.join(save_path, true_split, class_name, img_name+'.png')

                    wrong_files += 1

                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)

            elif tf_split == 'test':
                if true_split == 'train':
                    src_path = os.path.join(save_path, tf_split, class_name, img_name+'.png')
                    dst_path = os.path.join(save_path, true_split, class_name, img_name+'.png')

                    wrong_files += 1

                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)

        print(class_name, wrong_files)    


if __name__ == '__main__':
    data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM'

    mass_train_csv = os.path.join(data_root, 'mass_case_description_train_set.csv')
    mass_test_csv = os.path.join(data_root, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(data_root, 'calc_case_description_train_set.csv')
    calc_test_csv = os.path.join(data_root, 'calc_case_description_test_set.csv')


    # check_cbis_ddsm(mass_train_csv)
    # check_cbis_ddsm(mass_test_csv)

    # check_cbis_ddsm(calc_train_csv)
    # check_cbis_ddsm(calc_test_csv)

    #################################################################################
    tfds_data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM_tfds/val'
    check_cbis_ddsm_tfds(mass_train_csv, mass_test_csv,
                         calc_train_csv, calc_test_csv,
                         tfds_data_root, tf_split='val')

    tfds_data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM_tfds/train'
    check_cbis_ddsm_tfds(mass_train_csv, mass_test_csv,
                         calc_train_csv, calc_test_csv,
                         tfds_data_root, tf_split='train')

    tfds_data_root = '/home/hqvo2/Projects/Breast_Cancer/data/CBIS_DDSM_tfds/test'
    check_cbis_ddsm_tfds(mass_train_csv, mass_test_csv,
                         calc_train_csv, calc_test_csv,
                         tfds_data_root, tf_split='test')
