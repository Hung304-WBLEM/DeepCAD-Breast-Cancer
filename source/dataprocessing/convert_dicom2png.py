import pydicom
import cv2
import glob
import os

if __name__ == '__main__':
    data_root = '/data/hqvo2/reorganize_CBIS-DDSM'

    # mamm_img = pydicom.dcmread(dicom_path).pixel_array
    # cv2.imwrite(save_path, mamm_img)

    for idx, dcm_path in enumerate(glob.glob(os.path.join(data_root, '**/**/**', '*.dcm'))):
        print(idx)
        dcm_img = pydicom.dcmread(dcm_path).pixel_array
        file_path, ext = os.path.splitext(dcm_path)
        png_path = file_path + '.png'

        cv2.imwrite(png_path, dcm_img)
        




