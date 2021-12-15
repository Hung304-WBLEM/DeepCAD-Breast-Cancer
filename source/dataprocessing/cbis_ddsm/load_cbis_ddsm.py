import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import os
import glob

# Build dataset
# builder = tfds.builder("curated_breast_imaging_ddsm", data_dir = "/data/hqvo2/tf_CBIS-DDSM/")
# config = tfds.download.DownloadConfig(extract_dir="/data/hqvo2/ext_CBIS-DDSM", manual_dir="/data/hqvo2/reorganize_CBIS-DDSM")
# builder.download_and_prepare(download_dir = "/data/hqvo2/download_CBIS-DDSM", download_config = config)

# Iterate dataset
save_root = './save_dir/test'
ds = tfds.load('curated_breast_imaging_ddsm', split='test', data_dir='/data/hqvo2/tf_CBIS-DDSM')

classes = [ 'BACKGROUND', 'BENIGN_CALCIFICATION', 'BENIGN_MASS', 
            'MALIGNANT_CALCIFICATION', 'MALIGNANT_MASS']

for idx, example in enumerate(ds):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    print(idx, end=' ')

    sample_id = example['id'].numpy().decode('utf-8')
    image = example["image"].numpy()
    label = example["label"].numpy()

    save_dir = os.path.join(save_root, classes[label])
    os.makedirs(save_dir, exist_ok=True)

    sample_id = '#'.join(sample_id.split('/'))
    save_path = os.path.join(save_dir, sample_id + '.png')
    cv2.imwrite(save_path, image)


