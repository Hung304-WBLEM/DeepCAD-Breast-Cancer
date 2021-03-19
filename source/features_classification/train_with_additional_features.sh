cbis_ddsm_mass_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass_features_classification'
# python train_with_additional_features.py -d mass_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_mass_features_root}/mass_pathology_with_additional_features_r50_b32_e100_adam_wc

# python train_with_additional_features.py -d mass_pathology_clean \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_mass_features_root}/mass_pathology_clean_with_additional_features_r50_b32_e100_adam_wc
cbis_ddsm_calc_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_calc_features_classification'
# python train_with_additional_features.py -d calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_calc_features_root}/calc_pathology_with_additional_features_r50_b32_e100_adam_wc

# python train_with_additional_features.py -d calc_pathology_clean \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_calc_features_root}/calc_pathology_clean_with_additional_features_r50_b32_e100_adam_wc


cbis_ddsm_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification'
# python train_with_additional_features.py -d four_classes_mass_calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_r50_b32_e100_adam_wc

# python train_with_additional_features.py -d four_classes_mass_calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_uncertainty0.5_r50_b32_e100_adam_wc

# python train_with_additional_features.py -d four_classes_mass_calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -u 0.5 \
#        -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_uncertainty0.5_r50_b32_e100_adam_wc
# 
# python train_with_additional_features.py -d four_classes_mass_calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -u 0.7 \
#        -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_uncertainty0.7_r50_b32_e100_adam_wc
# 
# python train_with_additional_features.py -d four_classes_mass_calc_pathology \
#        -m resnet50 \
#        -b 32 -e 100 \
#        -opt adam -wc\
#        -u 0.8 \
#        -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_uncertainty0.8_r50_b32_e100_adam_wc

python train_with_additional_features.py -d four_classes_mass_calc_pathology \
       -m resnet50 \
       -b 32 -e 100 \
       -opt adam -wc\
       -u 0.9 \
       -s ${cbis_ddsm_features_root}/four_classes_mass_calc_pathology_with_additional_features_uncertainty0.9_r50_b32_e100_adam_wc
