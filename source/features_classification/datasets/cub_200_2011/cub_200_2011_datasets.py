import os
import json
from os.path import join

import numpy as np
import scipy
import scipy.misc
import torch
import pandas as pd
import matplotlib.pyplot as plt

from scipy import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

from features_classification.train.train_utils import compute_classes_weights

CUB_ROOT = '/home/hqvo2/Projects/Breast_Cancer/data/CUB_200_2011'


def get_CUB_classes(root):
    cub_classes = []
    with open(os.path.join(root, 'classes.txt'), 'r') as fin:
          for line in fin:
              id, class_name = line.strip().split()              
              cub_classes.append(class_name)

    return np.array(cub_classes)


class CUB_Dataset():
    '''Ref: https://github.com/TACJu/TransFG'''
    
    classes = get_CUB_classes(CUB_ROOT)

    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:            
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'crop_images', train_file.replace('jpg', 'png'))) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [os.path.join(self.root, 'crop_images', x.replace('jpg', 'png')) for x in train_file_list[:data_len]]
        if not self.is_train:            
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'crop_images', test_file.replace('jpg', 'png'))) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [os.path.join(self.root, 'crop_images', x.replace('jpg', 'png')) for x in test_file_list[:data_len]]

# Balanced Training Data
#    def get_classes_weights(self):
#        classes_weights = compute_classes_weights(
#          data_root=os.path.join(self.root, 'images'),
#          classes_names=CUB.classes
#        )
#        return classes_weights

    def __getitem__(self, index):
        if self.is_train:          
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        # return img, target
        return {'image': img, 'label': target, 'img_path': imgname}

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
