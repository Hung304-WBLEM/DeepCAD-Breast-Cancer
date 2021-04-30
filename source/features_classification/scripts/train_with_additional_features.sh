module load cudatoolkit/10.1

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'

mkdir -p ${four_classes_mass_calc_pathology_save_root}

cd ..


# Four classes Mass Calcification Pathology
python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
