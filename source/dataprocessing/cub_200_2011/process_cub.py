import cv2
import os
import glob

from config.cfg_loader import proj_paths_json


def crop_roi(cub_root):
    images_path = []
    with open(os.path.join(cub_root, 'images.txt'), 'r') as fin:
        for line in fin:
            images_path.append(line.strip().split()[1])

    bboxes = []
    with open(os.path.join(cub_root, 'bounding_boxes.txt'), 'r') as fin:
        for line in fin:
            _, x, y, width, height = line.strip().split()
            bbox = [x, y , width, height]
            bbox = [int(float(el)) for el in bbox]
            bboxes.append(bbox)

    save_root = os.path.join(cub_root, 'crop_images')
    os.makedirs(save_root, exist_ok=True)
    assert len(images_path) == len(bboxes)
    for idx, (img_path, bbox) in enumerate(zip(images_path, bboxes)):
        img_filepath = os.path.join(cub_root, 'images', img_path)

        class_name, img_filename = img_path.split('/')
        img = cv2.imread(img_filepath)

        x, y, w, h = bbox
        x_, y_ = x+w-1, y+h-1
        crop_img = img[y:y_, x:x_] 


        save_dir = os.path.join(save_root, class_name)
        os.makedirs(save_dir, exist_ok=True)

        img_filename = img_filename.replace('jpg', 'png')
        cv2.imwrite(os.path.join(save_dir, img_filename), crop_img)



if __name__ == '__main__':
    cub_root = '/home/hqvo2/Projects/Breast_Cancer/data/CUB_200_2011'
    crop_roi(cub_root)
