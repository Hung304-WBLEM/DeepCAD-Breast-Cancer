import os
import glob
import pandas as pd
import shutil

def check_cbis_ddsm(csv, data_root, save_root):
    df = pd.read_csv(csv)

    for index, row in df.iterrows():
        print(index, end=' ')
        img_path = row['image file path']
        cropped_path = row['cropped image file path']
        roi_path = row['ROI mask file path']


        img_dir = img_path.split('/')[0]
        cropped_dir = cropped_path.split('/')[0]
        roi_dir = cropped_path.split('/')[0]


        save_img_dir = os.path.join(save_root, os.path.dirname(img_path))
        save_cropped_dir = os.path.join(save_root, os.path.dirname(cropped_path))
        save_roi_dir = os.path.join(save_root, os.path.dirname(roi_path))

        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_cropped_dir, exist_ok=True)
        os.makedirs(save_roi_dir, exist_ok=True)

        print(save_img_dir)
        print(save_cropped_dir)
        print(roi_path)
        print('-'* 20)

        for path in glob.glob(os.path.join(data_root, img_dir, '**/**/000000.dcm')):
            filename = os.path.basename(path)
            shutil.copy(path, os.path.join(save_img_dir, filename))

        if len(glob.glob(os.path.join(data_root, roi_dir, '**/**/000001.dcm'))) == 0:
            for path in glob.glob(os.path.join(data_root, cropped_dir, '**/**/*.dcm')):
                filename = os.path.basename(path)
                if 'cropped' in path:
                    shutil.copy(path, os.path.join(save_cropped_dir, filename)) 
                elif 'ROI' in path:
                    shutil.copy(path, os.path.join(save_roi_dir, filename))
                else:
                    raise ValueError
        else:
            for path in glob.glob(os.path.join(data_root, cropped_dir, '**/**/000000.dcm')):
                filename = os.path.basename(path)
                shutil.copy(path, os.path.join(save_cropped_dir, filename))

            for path in glob.glob(os.path.join(data_root, roi_dir, '**/**/000001.dcm')):
                filename = os.path.basename(path)
                shutil.copy(path, os.path.join(save_roi_dir, filename))

        #if len(glob.glob(os.path.join(data_root, img_dir, '**/**/000000.dcm'))) == 0:
        #    fout.write(img_dir + '\n')
        #if len(glob.glob(os.path.join(data_root, cropped_dir, '**/**/*.dcm'))) < 2:
        #    fout.write(cropped_dir + '\n')
        #if len(glob.glob(os.path.join(data_root, roi_dir, '**/**/000001.dcm'))) == 0:
        #    fout.write(roi_dir + '\n')


    

if __name__ == '__main__':
    data_root = '/data/hqvo2/CBIS-DDSM'
    mass_train_csv = os.path.join(data_root, 'mass_case_description_train_set.csv')
    mass_test_csv = os.path.join(data_root, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(data_root, 'calc_case_description_train_set.csv')
    calc_test_csv = os.path.join(data_root, 'calc_case_description_test_set.csv')

    save_root = '/data/hqvo2/reorganize_CBIS-DDSM'


    # check_cbis_ddsm(mass_train_csv, data_root, save_root)
    check_cbis_ddsm(mass_test_csv, data_root, save_root)
    check_cbis_ddsm(calc_train_csv, data_root, save_root)
    check_cbis_ddsm(calc_test_csv, data_root, save_root)
