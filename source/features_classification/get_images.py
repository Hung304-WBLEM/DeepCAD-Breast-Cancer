import os
import glob
from shutil import copyfile

config_without_feats = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_r50_b32_e100_adam_drop05'
config_with_feats = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_with_additional_features_r50_b32_e100_adam_wc'

result_without_feats = []
with open(os.path.join(config_without_feats, 'test_result.txt'), 'r') as f:
    for line in f:
        path, label, pred = line.rstrip().split()
        result_without_feats.append((path, label, pred))
sorted(result_without_feats, key=lambda x: x[0])

result_with_feats = []
with open(os.path.join(config_with_feats, 'test_result.txt'), 'r') as f:
    for line in f:
        path, label, pred = line.rstrip().split()
        result_with_feats.append((path, label, pred))

sorted(result_with_feats, key=lambda x: x[0])


# True in 'without' but False in 'with'
save_root = os.path.join(config_without_feats, 'test_images')
os.makedirs(save_root, exist_ok=True)
true_in_without_but_false_in_with = []
for ret_without, ret_with in zip(result_without_feats, result_with_feats):
    if ret_without[2] == ret_without[1] and ret_with[2] != ret_with[1]:
        true_in_without_but_false_in_with.append((ret_without[0], ret_without[2], ret_with[2]))

for path, pred_without, pred_with in true_in_without_but_false_in_with:
    save_path = os.path.join(save_root, pred_without, pred_with)
    os.makedirs(save_path, exist_ok=True)
    img_filename = os.path.basename(path)
    copyfile(path, os.path.join(save_path, img_filename))



# True in 'with' but False in 'without'
save_root = os.path.join(config_with_feats, 'test_images')
os.makedirs(save_root, exist_ok=True)
true_in_with_but_false_in_without = []
for ret_without, ret_with in zip(result_without_feats, result_with_feats):
    if ret_with[2] == ret_with[1] and ret_without[2] != ret_without[1]:
        true_in_with_but_false_in_without.append((ret_with[0], ret_with[2], ret_without[2]))

for path, pred_with, pred_without in true_in_with_but_false_in_without:
    save_path = os.path.join(save_root, pred_with, pred_without)
    os.makedirs(save_path, exist_ok=True)
    img_filename = os.path.basename(path)
    copyfile(path, os.path.join(save_path, img_filename))
