module load cudatoolkit/10.1

cbis_ddsm_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification'

python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_features_root}/test

# python train.py -d "four_classes_mass_calc_pathology_2048x2048-crop_zero-pad" -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_2048x2048-crop_zero-pad_r50_b32_e100_adam_wc

# python train.py -d "four_classes_mass_calc_pathology" -m resnet50 -b 32 -e 600 -opt sgd -wc -s ${cbis_ddsm_features_root}/test

