import os
import glob
import pandas as pd

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
            if len(glob.glob(os.path.join(data_root, roi_dir, '**/**/000001.dcm'))) == 0:
                fout.write(roi_dir + '\n')


    

if __name__ == '__main__':
    data_root = '/data/hqvo2/CBIS-DDSM'
    mass_train_csv = os.path.join(data_root, 'mass_case_description_train_set.csv')
    mass_test_csv = os.path.join(data_root, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(data_root, 'calc_case_description_train_set.csv')
    calc_test_csv = os.path.join(data_root, 'calc_case_description_test_set.csv')


    check_cbis_ddsm(mass_train_csv)
    check_cbis_ddsm(mass_test_csv)

    check_cbis_ddsm(calc_train_csv)
    check_cbis_ddsm(calc_test_csv)
