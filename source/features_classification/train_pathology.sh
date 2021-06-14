module load cudatoolkit/10.1

cbis_ddsm_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification'

# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 200 -opt adam -wc -ws -s ${cbis_ddsm_features_root}/test_ws_200eps
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 200 -opt adam -wc -s ${cbis_ddsm_features_root}/test_200eps
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -opt adam -wc -ws -s ${cbis_ddsm_features_root}/test_normal_224_ws
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_features_root}/test_normal_224

python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 3 -i 224 --opt adam --wc --ws -s ${cbis_ddsm_features_root}/temp

# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 200 -opt adam -wc -s ${cbis_ddsm_features_root}/test_200eps_clahe
# python train.py -d "four_classes_mass_calc_pathology_512x512-crop_zero-pad" -m resnet50 -b 32 -e 200 -opt adam -wc -ws -s ${cbis_ddsm_features_root}/test_512x512-crop_zero-pad_ws
# python train.py -d "four_classes_mass_calc_pathology_512x512-crop_zero-pad" -m resnet50 -b 32 -e 200 -opt adam -wc -s ${cbis_ddsm_features_root}/test_512x512-crop_zero-pad

# python train.py -d "four_classes_mass_calc_pathology_512x512-crop_zero-pad" -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_512x512-crop_zero-pad_r50_b32_e100_adam_wc_ws_best-val-loss

# python train.py -d "four_classes_mass_calc_pathology" -m resnet50 -b 32 -e 600 -opt sgd -wc -s ${cbis_ddsm_features_root}/test

