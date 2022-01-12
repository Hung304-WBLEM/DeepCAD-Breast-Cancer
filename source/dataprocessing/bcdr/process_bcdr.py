import mmcv
import numpy as np
import os
import pandas as pd
import glob
import cv2

from config.cfg_loader import proj_paths_json
from pycocotools import mask as coco_api_mask
from dataprocessing.random_patches_sampling import _sample_positive_patches
from dataprocessing.random_patches_sampling import _sample_negative_patches
from dataprocessing.cbis_ddsm.remove_blank_background import remove_background_images

def area(_mask):
    rle = coco_api_mask.encode(np.asfortranarray(_mask))
    area = coco_api_mask.area(rle)
    return area

def get_random_crops(data_root, background_save_root):
    os.makedirs(background_save_root, exist_ok=True)

    abnormal_masks = []
    abnormal_areas = []
    for mamm_path in mmcv.track_iter_progress(glob.glob(os.path.join(data_root, '**', '**', '*.tif'))):
        mamm_img = mmcv.imread(mamm_path)
        filename, _ = os.path.splitext(os.path.basename(mamm_path))

        for patch_id, neg_patch in enumerate(_sample_negative_patches(mamm_img,
                                                                    abnormal_masks,
                                                                    abnormal_areas,
                                                                    (224, 224))):
            save_path = os.path.join(background_save_root,
                                     filename.strip() + f'_background_{patch_id}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cv2.imwrite(save_path, neg_patch)


def get_bcdr_lesion_pathology(data_root,
                              mass_pathology_save_root,
                              calc_pathology_save_root,
                              microcalc_pathology_save_root,
                              mass_calc_pathology_save_root,
                              mass_microcalc_pathology_save_root,
                              calc_microcalc_pathology_save_root,
                              background_save_root=None,
                              patch_ext='center'):
    
    outline_csv_path = glob.glob(os.path.join(data_root, '*outlines.csv'))[0]
    outline_csv = pd.read_csv(outline_csv_path)

    prev_filename = None
    abnormal_masks = []
    abnormal_areas = []

    for index, row in outline_csv.iterrows():
        lesion_id = row['lesion_id']
        filename = row['image_filename']
        x_points = row['lw_x_points']
        y_points = row['lw_y_points']

        img_name = os.path.join(data_root, filename.strip())
        mamm_img = mmcv.imread(os.path.join(data_root, 'AllPNGs', img_name))
        resized_mamm_img = cv2.resize(mamm_img, (896, 1152))
        height, width = mamm_img.shape[:2]

        is_mass = row['mammography_nodule']
        is_calc = row['mammography_calcification']
        is_microcalc = row['mammography_microcalcification']

        pathology_label = row['classification']

        if pathology_label.strip() in ['Malignant', 'Malign']:
            label = 'BENIGN'
            cat_id = 1
        elif pathology_label.strip() == 'Benign':
            label = 'MALIGNANT'
            cat_id = 0
        else:
            print(pathology_label)
            raise ValueError

        # is_axillary_adenopathy = row['mammography_axillary_adenopathy']
        # is_architectural_distortion = row['mammography_architectural_distortion']
        # is_stroma_distortion = row['mammography_stroma_distortion']

        if is_mass == 0 and is_calc == 0 and is_microcalc == 0:
            # remove several rows in F01 set that is neither mass or calc
            continue


        x_points = [int(el) for el in x_points.split()]
        y_points = [int(el) for el in y_points.split()]

        xy_seg_poly = []
        draw_xy_seg_poly = []

        for x, y in zip(x_points, y_points):
            xy_seg_poly.extend([x, y])
            draw_xy_seg_poly.append([x, y])
        
        seg_poly = [xy_seg_poly]
        flat_seg_poly = [el for sublist in seg_poly for el in sublist]
        px = flat_seg_poly[::2]
        py = flat_seg_poly[1::2]
        x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

        patch_width = x_max - x_min + 1
        patch_height = y_max - y_min + 1

        # Skip too small images
        if patch_width * patch_height < 32*32:
            continue


        if patch_ext == 'random':
            mask_img = np.zeros([height, width], dtype=np.uint8)
            cv2.fillPoly(mask_img, [np.array(draw_xy_seg_poly, np.int32)], 255)
            resized_mask_img = cv2.resize(mask_img, (896, 1152))

            mask_area = np.sum(resized_mask_img > 0)

            abnormal_masks.append(resized_mask_img)
            abnormal_areas.append(mask_area)


        if patch_ext == 'center':
            center_x = (x_min + x_max)/2.0
            center_y = (y_min + y_max)/2.0

            patch_size = max(patch_width, patch_height)

            new_x_min = max(0, int(center_x - patch_size/2))
            new_y_min = max(0, int(center_y - patch_size/2))
            new_x_max = min(int(width), int(center_x + patch_size/2))
            new_y_max = min(int(height), int(center_y + patch_size/2))


        if patch_ext == 'exact':
            lesion_patch = mamm_img[y_min:(y_max+1), x_min:(x_max+1)]
        elif patch_ext == 'center':
            lesion_patch = mamm_img[new_y_min:(new_y_max+1), new_x_min:(new_x_max+1)]

            pad_width_size = max(patch_size - (new_x_max - new_x_min), 0)
            pad_height_size = max(patch_size - (new_y_max - new_y_min), 0)

            lesion_patch = np.pad(lesion_patch,
                                [(int(pad_height_size//2),
                                    int(pad_height_size - int(pad_height_size//2))),
                                    (int(pad_width_size//2),
                                    int(pad_width_size - int(pad_width_size//2))),
                                    (0, 0)], 'constant')

        if is_mass + is_calc + is_microcalc == 1:
            if is_mass:
                save_root = mass_pathology_save_root
            elif is_calc:
                save_root = calc_pathology_save_root
            elif is_microcalc:
                save_root = microcalc_pathology_save_root
        elif is_mass + is_calc + is_microcalc == 2:
            if is_mass + is_calc == 2:
                save_root = mass_calc_pathology_save_root
            elif is_mass + is_microcalc == 2:
                save_root = mass_microcalc_pathology_save_root
            elif is_calc + is_microcalc == 2:
                save_root = calc_microcalc_pathology_save_root


        if cat_id == 0:
            save_root = os.path.join(save_root, 'MALIGNANT')
        elif cat_id == 1:
            save_root = os.path.join(save_root, 'BENIGN')

        filename = os.path.splitext(os.path.basename(filename))[0]

        if patch_ext in ['exact', 'center']:
            save_path = os.path.join(save_root, filename.strip() + f'_{lesion_id}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cv2.imwrite(save_path, lesion_patch)
        elif patch_ext == 'random':
            for patch_id, pos_patch in enumerate(_sample_positive_patches(resized_mamm_img,
                                                                          resized_mask_img,
                                                                          mask_area,
                                                                          (224, 224))):
                save_path = os.path.join(save_root,
                                         filename.strip() + f'_{lesion_id}_patch_{patch_id}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                cv2.imwrite(save_path, pos_patch)

            if filename != prev_filename:
                if prev_filename is not None:
                    os.makedirs(background_save_root, exist_ok=True)
                    for patch_id, neg_patch in enumerate(_sample_negative_patches(resized_mamm_img,
                                                                                abnormal_masks,
                                                                                abnormal_areas,
                                                                                (224, 224))):
                        save_path = os.path.join(background_save_root,
                                                 filename.strip() + f'_{lesion_id}_background_{patch_id}.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        cv2.imwrite(save_path, neg_patch)
                prev_filename = filename
                abnormal_masks = []
                abnormal_areas = []



if __name__ == '__main__':
    data_root = proj_paths_json['DATA']['root']
    bcdr_root = os.path.join(
        data_root, proj_paths_json['DATA']['BCDR']['root'])

    film_data = proj_paths_json['DATA']['BCDR']['film']
    film_data_root = os.path.join(bcdr_root, film_data['root'])
    digital_data = proj_paths_json['DATA']['BCDR']['digital']
    digital_data_root = os.path.join(bcdr_root, digital_data['root'])


    data_name_list = ['BCDR-D01_dataset', 'BCDR-D02_dataset',
                      'BCDR-F01_dataset', 'BCDR-F02_dataset', 'BCDR-F03_dataset']
    for data_name in data_name_list:
        data_type = data_name.split('_')[0].split('-')[1][0]

        if data_type == 'F':
            save_root = film_data_root
            save_data = film_data
        elif data_type == 'D':
            save_root = digital_data_root
            save_data = digital_data
        
        get_bcdr_lesion_pathology(os.path.join(bcdr_root, data_name),
                                mass_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['mass_pathology']),
                                calc_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['calc_pathology']),
                                microcalc_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['microcalc_pathology']),
                                mass_calc_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['mass_calc_pathology']),
                                mass_microcalc_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['mass_microcalc_pathology']),
                                calc_microcalc_pathology_save_root=\
                                os.path.join(save_root, data_name,
                                            save_data['calc_microcalc_pathology'])
                                )


        get_bcdr_lesion_pathology(os.path.join(bcdr_root, data_name),
                                  mass_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_mass_pathology']),
                                  calc_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_calc_pathology']),
                                  microcalc_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_microcalc_pathology']),
                                  mass_calc_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_mass_calc_pathology']),
                                  mass_microcalc_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_mass_microcalc_pathology']),
                                  calc_microcalc_pathology_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['aug_calc_microcalc_pathology']),
                                  background_save_root=\
                                  os.path.join(save_root, data_name,
                                               save_data['background']['bg_tfds']),
                                  patch_ext='random'
                                  )

        remove_background_images(os.path.join(save_root, data_name,
                                            save_data['background']['bg_tfds']))


    # Get background crops from BCDR-DN01_dataset
    get_random_crops(data_root=os.path.join(bcdr_root, 'BCDR-DN01_dataset'),
                     background_save_root=os.path.join(digital_data_root,
                                                       'BCDR-DN01_dataset',
                                                       digital_data['background']['bg_tfds']
                                                       ))
    remove_background_images(os.path.join(digital_data_root,
                                          'BCDR-DN01_dataset',
                                          digital_data['background']['bg_tfds']))
