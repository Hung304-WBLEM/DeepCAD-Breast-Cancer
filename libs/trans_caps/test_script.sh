module load cudatoolkit/10.1

# python train.py --dn four_classes_mass_calc_pathology --nc 4 --rv resnet32 -b 16 -e 500 --sd /home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_transcaps_4convcaps_r32_e500_b16_224x224
# python train.py --dn four_classes_mass_calc_pathology --nc 4 -b 32 -e 500 --sd /home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_transcaps_e500_b32_256x256

python inference.py --dn four_classes_mass_calc_pathology --nc 4 --rv resnet32 -b 16 --lp /home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_transcaps_4convcaps_r32_e500_b16_224x224/20210317_005722_TR_resnet_four_classes_mass_calc_pathology/models 
# python inference.py --dn four_classes_mass_calc_pathology --nc 4 --rv resnet32 -b 32 --lp /home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification/four_classes_mass_calc_pathology_transcaps_r32_e500_b32_256x256/20210315_081323_TR_resnet_four_classes_mass_calc_pathology/models 
