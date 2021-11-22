module load cudatoolkit/10.1

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'
four_classes_mass_calc_pathology_histeq_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology_histeq'

mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
mass_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion'
mass_breast_density_image_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_image'

calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'
calc_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion'
calc_breast_density_image_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_image'

# with centercrop
mass_shape_comb_feats_omit_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit_centercrop'
mass_margins_comb_feats_omit_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit_centercrop'
mass_breast_density_lesion_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion_centercrop'

calc_type_comb_feats_omit_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit_centercrop'
calc_dist_comb_feats_omit_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit_centercrop'
calc_breast_density_lesion_centercrop_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion_centercrop'

# with segmentation
mass_shape_comb_feats_omit_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit_segm'
mass_margins_comb_feats_omit_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit_segm'
mass_breast_density_lesion_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion_segm'

calc_type_comb_feats_omit_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit_segm'
calc_dist_comb_feats_omit_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit_segm'
calc_breast_density_lesion_segm_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion_segm'

# with mask
mass_shape_comb_feats_omit_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit_mask'
mass_margins_comb_feats_omit_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit_mask'
mass_breast_density_lesion_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion_mask'

calc_type_comb_feats_omit_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit_mask'
calc_dist_comb_feats_omit_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit_mask'
calc_breast_density_lesion_mask_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion_mask'

mkdir -p ${four_classes_mass_calc_pathology_save_root}
mkdir -p ${four_classes_mass_calc_pathology_histeq_save_root}
mkdir -p ${mass_shape_comb_feats_omit_save_root}
mkdir -p ${mass_margins_comb_feats_omit_save_root}
mkdir -p ${mass_breast_density_lesion_save_root}
mkdir -p ${calc_type_comb_feats_omit_save_root}
mkdir -p ${calc_dist_comb_feats_omit_save_root}
mkdir -p ${calc_breast_density_lesion_save_root}
mkdir -p ${mass_shape_comb_feats_omit_segm_save_root}
mkdir -p ${mass_margins_comb_feats_omit_segm_save_root}
mkdir -p ${mass_breast_density_lesion_segm_save_root}
mkdir -p ${calc_type_comb_feats_omit_segm_save_root}
mkdir -p ${calc_dist_comb_feats_omit_segm_save_root}
mkdir -p ${calc_breast_density_lesion_segm_save_root}
mkdir -p ${mass_shape_comb_feats_omit_mask_save_root}
mkdir -p ${mass_margins_comb_feats_omit_mask_save_root}
mkdir -p ${mass_breast_density_lesion_mask_save_root}
mkdir -p ${calc_type_comb_feats_omit_mask_save_root}
mkdir -p ${calc_dist_comb_feats_omit_mask_save_root}
mkdir -p ${calc_breast_density_lesion_mask_save_root}

cd ..

# Four classes Mass Calcification Pathology
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 256 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_256x256_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 512 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_512x512_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 1024 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_1024x1024_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 --third_stage_freeze 50 -b 32 -e 100 -i 224 --opt adam -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_"$(LC_TIME="EN.UTF-8" date)"


# Four classes Mass Calcification Pathology (for plotting learning curve)
# for i in {1..9}
# do
#     python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done

# python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_shape_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${mass_shape_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_margins_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${mass_margins_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py -d mass_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d mass_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${mass_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py --njobs 8 -d mass_breast_density_image -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_breast_density_image_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py --njobs 8 -d mass_breast_density_image -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${mass_breast_density_image_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done

# python train.py -d calc_type_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_type_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d calc_type_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${calc_type_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py -d calc_dist_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_dist_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d calc_dist_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${calc_dist_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py -d calc_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py -d calc_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${calc_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done
# python train.py --njobs 8 -d calc_breast_density_image -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_breast_density_image_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# for i in {1..9}
# do
#     python train.py --njobs 8 -d calc_breast_density_image -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${calc_breast_density_image_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
# done

# python train.py -d four_classes_mass_calc_pathology_histeq -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_histeq_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology_histeq -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_histeq_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Train with segmentation
# python train.py -d mass_shape_comb_feats_omit_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_shape_comb_feats_omit_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_margins_comb_feats_omit_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_margins_comb_feats_omit_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_breast_density_lesion_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_breast_density_lesion_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d calc_type_comb_feats_omit_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_type_comb_feats_omit_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d calc_dist_comb_feats_omit_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_dist_comb_feats_omit_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d calc_breast_density_lesion_segm -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_breast_density_lesion_segm_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Train with mask
# python train.py -d mass_shape_comb_feats_omit_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_shape_comb_feats_omit_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_margins_comb_feats_omit_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_margins_comb_feats_omit_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_breast_density_lesion_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_breast_density_lesion_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d calc_type_comb_feats_omit_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_type_comb_feats_omit_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d calc_dist_comb_feats_omit_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_dist_comb_feats_omit_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d calc_breast_density_lesion_mask -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_breast_density_lesion_mask_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Train with BCE Loss
# python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --crt bce -s ${mass_shape_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --crt bce -s ${mass_margins_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d calc_type_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --crt bce -s ${calc_type_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d calc_dist_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --crt bce -s ${calc_dist_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Train using albumentations for augmentation
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt ce --aug_type albumentations -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_aug-albumentations_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 200 -i 224 --wc --ws --opt adam --crt ce --aug_type albumentations --is_fourth_stage -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e200_4stages_224x224_adam_wc_ws_aug-albumentations_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt bce --aug_type albumentations -s ${mass_shape_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_aug-albumentations_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt bce --aug_type albumentations -s ${mass_margins_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_bce_wc_ws_aug-albumentations_"$(LC_TIME="EN.UTF-8" date)"

# Train using augmix for augmentation
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt ce --aug_type augmix -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_aug-augmix_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 1000 -i 224 --wc --ws --opt adam --crt ce --aug_type augmix -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e1000_224x224_adam_wc_ws_aug-augmix_"$(LC_TIME="EN.UTF-8" date)"
# python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt ce --aug_type augmix -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_aug-augmix-allops_"$(LC_TIME="EN.UTF-8" date)"

# Train with dilated convolution
# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_2nd --rnet_dil_3rd --rnet_dil_4th\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_2nd --rnet_dil_3rd --rnet_dil_4th\
#        -b 32 \
#        -e 100 -i 256 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_256x256_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_2nd --rnet_dil_3rd\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_2nd\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_3rd --rnet_dil_4th\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py -d four_classes_mass_calc_pathology \
#        -m dilated_resnet50 \
#        --rnet_dil_4th\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        -s ${four_classes_mass_calc_pathology_save_root}/dilated_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# EfficientNet
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'efficientnet-b0' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \
#        -s ${four_classes_mass_calc_pathology_save_root}/efficientnet-b0_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'efficientnet-b4' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 415 \
#        --second_stage_freeze 360 \
#        -s ${four_classes_mass_calc_pathology_save_root}/efficientnet-b4_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'efficientnet-b0' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \
#        -s ${mass_shape_comb_feats_omit_save_root}/efficientnet-b0_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'efficientnet-b4' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \
#        -s ${mass_shape_comb_feats_omit_save_root}/efficientnet-b4_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # EfficientNet V2
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_s_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 449 \
#        --second_stage_freeze 381 \
#        -s ${four_classes_mass_calc_pathology_save_root}/tf-efficientnetv2-s-in21ft1k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_m_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 449 \
#        --second_stage_freeze 381 \
#        -s ${four_classes_mass_calc_pathology_save_root}/tf-efficientnetv2-m-in21ft1k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_s_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \
#        -s ${mass_shape_comb_feats_omit_save_root}/tf-efficientnetv2-s-in21ft1k_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_m_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \
#        -s ${mass_shape_comb_feats_omit_save_root}/tf-efficientnetv2-m-in21ft1k_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# EfficientNet NoisyStudent
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'tf_efficientnet_b0_ns'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 210 \
#        --second_stage_freeze 142 \
#        -s ${four_classes_mass_calc_pathology_save_root}/tf-efficientnet-b0-ns_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'tf_efficientnet_b4_ns'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 415 \
#        --second_stage_freeze 360 \
#        -s ${four_classes_mass_calc_pathology_save_root}/tf-efficientnet-b4-ns_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'tf_efficientnet_b0_ns'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 142 \
#        -s ${mass_shape_comb_feats_omit_save_root}/tf-efficientnet-b0-ns_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'tf_efficientnet_b4_ns'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 415 \
#        --second_stage_freeze 360 \
#        -s ${mass_shape_comb_feats_omit_save_root}/tf-efficientnet-b4-ns_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# Vision Transformer
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws \
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-base-patch16-384_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_base_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-base-patch16-224-in21k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_huge_patch14_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 391 \
#        --second_stage_freeze 339 \
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-huge-patch14-224-in21k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_large_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 293 \
#        --second_stage_freeze 243 \
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-large-patch16-224-in21k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Mass shape
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-224-in21k_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_huge_patch14_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 391 \
#        --second_stage_freeze 339 \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-huge-patch14-224-in21k_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_large_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 293 \
#        --second_stage_freeze 243 \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-large-patch16-224-in21k_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Mass Margins
# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_margins_comb_feats_omit_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 128 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_margins_comb_feats_omit_save_root}/vit-base-patch16-224_b128_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_margins_comb_feats_omit_save_root}/vit-base-patch16-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Mass Breast Density Image
# python train.py --save \
#        -d mass_breast_density_image \
#        --njobs 10 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_breast_density_image_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Calc Type
# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_type_comb_feats_omit_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_type_comb_feats_omit_save_root}/vit-base-patch16-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Calc Dist
# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_dist_comb_feats_omit_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_dist_comb_feats_omit_save_root}/vit-base-patch16-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Calc Breast Density Image
# python train.py --save \
#        -d calc_breast_density_image \
#        --njobs 10 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_breast_density_image_save_root}/vit-base-patch16-224_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# Convit
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'convit_base'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 177 \
#        --second_stage_freeze 123 \
#        -s ${four_classes_mass_calc_pathology_save_root}/convit-base_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'convit_base'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 177 \
#        --second_stage_freeze 123 \
#        -s ${mass_shape_comb_feats_omit_save_root}/convit-base_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Twins
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'twins_svt_base'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 381 \
#        --second_stage_freeze 327 \
#        -s ${four_classes_mass_calc_pathology_save_root}/twins-svt-base_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'twins_svt_base'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 381 \
#        --second_stage_freeze 327 \
#        -s ${mass_shape_comb_feats_omit_save_root}/twins-svt-base_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# Bit
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'resnetv2_101x1_bitm'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 303 \
#        --second_stage_freeze 246 \
#        -s ${four_classes_mass_calc_pathology_save_root}/resnetv2-101x1-bitm_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'resnetv2_101x1_bitm'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 303 \
#        --second_stage_freeze 246 \
#        -s ${mass_shape_comb_feats_omit_save_root}/resnetv2-101x1-bitm_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# T2T-ViT
# # T2T-ViT-14
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'T2T-ViT-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${four_classes_mass_calc_pathology_save_root}/T2T-ViT-14_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${mass_shape_comb_feats_omit_save_root}/T2T-ViT-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${mass_margins_comb_feats_omit_save_root}/T2T-ViT-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${calc_type_comb_feats_omit_save_root}/T2T-ViT-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${calc_dist_comb_feats_omit_save_root}/T2T-ViT-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # T2T-ViT-19
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'T2T-ViT-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 240 \
#        --second_stage_freeze 194 \
#        -s ${four_classes_mass_calc_pathology_save_root}/T2T-ViT-19_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 240 \
#        --second_stage_freeze 194 \
#        -s ${mass_shape_comb_feats_omit_save_root}/T2T-ViT-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 240 \
#        --second_stage_freeze 194 \
#        -s ${mass_margins_comb_feats_omit_save_root}/T2T-ViT-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 240 \
#        --second_stage_freeze 194 \
#        -s ${calc_type_comb_feats_omit_save_root}/T2T-ViT-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 240 \
#        --second_stage_freeze 194 \
#        -s ${calc_dist_comb_feats_omit_save_root}/T2T-ViT-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # T2T-ViT-14_384
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'T2T-ViT-14_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws \
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${four_classes_mass_calc_pathology_save_root}/T2T-ViT-14-384_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${mass_shape_comb_feats_omit_save_root}/T2T-ViT-14-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${mass_margins_comb_feats_omit_save_root}/T2T-ViT-14-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${calc_type_comb_feats_omit_save_root}/T2T-ViT-14-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT-14_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 185 \
#        --second_stage_freeze 128 \
#        -s ${calc_dist_comb_feats_omit_save_root}/T2T-ViT-14-384_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # T2T-ViT_t-14
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'T2T-ViT_t-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 181 \
#        --second_stage_freeze 124 \
#        -s ${four_classes_mass_calc_pathology_save_root}/T2T-ViT-t-14_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 181 \
#        --second_stage_freeze 124 \
#        -s ${mass_shape_comb_feats_omit_save_root}/T2T-ViT-t-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 181 \
#        --second_stage_freeze 124 \
#        -s ${mass_margins_comb_feats_omit_save_root}/T2T-ViT-t-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 181 \
#        --second_stage_freeze 124 \
#        -s ${calc_type_comb_feats_omit_save_root}/T2T-ViT-t-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-14'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 181 \
#        --second_stage_freeze 124 \
#        -s ${calc_dist_comb_feats_omit_save_root}/T2T-ViT-t-14_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # T2T-ViT_t-19
# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'T2T-ViT_t-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze 236 \
#        --second_stage_freeze 190 \
#        -s ${four_classes_mass_calc_pathology_save_root}/T2T-ViT-t-19_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 236 \
#        --second_stage_freeze 190 \
#        -s ${mass_shape_comb_feats_omit_save_root}/T2T-ViT-t-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 236 \
#        --second_stage_freeze 190 \
#        -s ${mass_margins_comb_feats_omit_save_root}/T2T-ViT-t-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 236 \
#        --second_stage_freeze 190 \
#        -s ${calc_type_comb_feats_omit_save_root}/T2T-ViT-t-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'T2T-ViT_t-19'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 236 \
#        --second_stage_freeze 190 \
#        -s ${calc_dist_comb_feats_omit_save_root}/T2T-ViT-t-19_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 256 \
#        --not_use_pretrained \
#        -e 4000 -i 224 --opt adam --wc --ws \
#        --first_stage_freeze -1 --first_stage_wd 0.1 --first_stage_lr 0.1\
#        --second_stage_freeze -1 --second_stage_wd 0.1 --second_stage_lr 0.1\
#        --third_stage_freeze -1 --third_stage_wd 0.1 --third_stage_lr 0.1\
#        -s ${four_classes_mass_calc_pathology_save_root}/vit-base-patch16-224_nopretrained_b256_e1000_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 256 \
#        --not_use_pretrained \
#        -e 1000 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze -1 --first_stage_wd 0.1 --first_stage_lr 0.1\
#        --second_stage_freeze -1 --second_stage_wd 0.1 --second_stage_lr 0.01\
#        --third_stage_freeze -1 --third_stage_wd 0.1 --third_stage_lr 0.001\
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-224_nopretrained_b256_e1000_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # CaiT
# ## cait_m36_384
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_m36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        --third_stage_freeze 300 \
#        -s ${mass_shape_comb_feats_omit_save_root}/cait-m36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_m36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        --third_stage_freeze 300 \
#        -s ${mass_margins_comb_feats_omit_save_root}/cait-m36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_m36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        --third_stage_freeze 300 \
#        -s ${calc_type_comb_feats_omit_save_root}/cait-m36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_m36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        --third_stage_freeze 300 \
#        -s ${calc_dist_comb_feats_omit_save_root}/cait-m36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# ## cait_s36_384
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_s36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        -s ${mass_shape_comb_feats_omit_save_root}/cait-s36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_s36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        -s ${mass_margins_comb_feats_omit_save_root}/cait-s36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_s36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        -s ${calc_type_comb_feats_omit_save_root}/cait-s36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'cait_s36_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 689 \
#        --second_stage_freeze 633 \
#        -s ${calc_dist_comb_feats_omit_save_root}/cait-s36-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # coat
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'coat_lite_small'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 245 \
#        --second_stage_freeze 195 \
#        -s ${mass_shape_comb_feats_omit_save_root}/coat-lite-small_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'coat_lite_small'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 245 \
#        --second_stage_freeze 195 \
#        -s ${mass_margins_comb_feats_omit_save_root}/coat-lite-small_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'coat_lite_small'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 245 \
#        --second_stage_freeze 195 \
#        -s ${calc_type_comb_feats_omit_save_root}/coat-lite-small_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'coat_lite_small'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 245 \
#        --second_stage_freeze 195 \
#        -s ${calc_dist_comb_feats_omit_save_root}/coat-lite-small_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # deit
# ## deit_base_patch16_224
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_shape_comb_feats_omit_save_root}/deit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_margins_comb_feats_omit_save_root}/deit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_type_comb_feats_omit_save_root}/deit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_dist_comb_feats_omit_save_root}/deit-base-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# ## deit_base_patch16_384
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_shape_comb_feats_omit_save_root}/deit-base-patch16-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${mass_margins_comb_feats_omit_save_root}/deit-base-patch16-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_type_comb_feats_omit_save_root}/deit-base-patch16-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'deit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        -s ${calc_dist_comb_feats_omit_save_root}/deit-base-patch16-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # pit
# ## pit_b_224
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 169 \
#        --second_stage_freeze 119 \
#        -s ${mass_shape_comb_feats_omit_save_root}/pit-b-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 169 \
#        --second_stage_freeze 119 \
#        -s ${mass_margins_comb_feats_omit_save_root}/pit-b-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 169 \
#        --second_stage_freeze 119 \
#        -s ${calc_type_comb_feats_omit_save_root}/pit-b-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 169 \
#        --second_stage_freeze 119 \
#        -s ${calc_dist_comb_feats_omit_save_root}/pit-b-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# ## pit_b_distilled_224
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_distilled_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 171 \
#        --second_stage_freeze 119 \
#        -s ${mass_shape_comb_feats_omit_save_root}/pit-b-distilled-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_distilled_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 171 \
#        --second_stage_freeze 119 \
#        -s ${mass_margins_comb_feats_omit_save_root}/pit-b-distilled-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_distilled_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 171 \
#        --second_stage_freeze 119 \
#        -s ${calc_type_comb_feats_omit_save_root}/pit-b-distilled-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'pit_b_distilled_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 171 \
#        --second_stage_freeze 119 \
#        -s ${calc_dist_comb_feats_omit_save_root}/pit-b-distilled-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"


# # tnt
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'tnt_s_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 348 \
#        --second_stage_freeze 290 \
#        -s ${mass_shape_comb_feats_omit_save_root}/tnt-s-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'tnt_s_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 348 \
#        --second_stage_freeze 290 \
#        -s ${mass_margins_comb_feats_omit_save_root}/tnt-s-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'tnt_s_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 348 \
#        --second_stage_freeze 290 \
#        -s ${calc_type_comb_feats_omit_save_root}/tnt-s-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'tnt_s_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 348 \
#        --second_stage_freeze 290 \
#        -s ${calc_dist_comb_feats_omit_save_root}/tnt-s-patch16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# # swin
# ## swin_base_patch4_window7_224
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window7_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${mass_shape_comb_feats_omit_save_root}/swin-base-patch4-window7-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window7_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${mass_margins_comb_feats_omit_save_root}/swin-base-patch4-window7-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window7_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${calc_type_comb_feats_omit_save_root}/swin-base-patch4-window7-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window7_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${calc_dist_comb_feats_omit_save_root}/swin-base-patch4-window7-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# ## swin_base_patch4_window12_384
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${mass_shape_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${mass_margins_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${calc_type_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        -s ${calc_dist_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# CrossViT
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'crossvit_18_240'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 371 \
#        --second_stage_freeze 319 \
#        -s ${mass_shape_comb_feats_omit_save_root}/crossvit-18-240_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# best ckpt (val acc)
# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-224_b32_e100_bestacc_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window7_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_shape_comb_feats_omit_save_root}/swin-base-patch4-window7-224_b32_e100_bestacc_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_shape_comb_feats_omit_save_root}/vit-base-patch16-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_shape_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_shape_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_margins_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${calc_type_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${calc_dist_comb_feats_omit_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# center-crop
# python train.py --save \
#        -d mass_shape_comb_feats_omit_centercrop \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_shape_comb_feats_omit_centercrop_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d mass_margins_comb_feats_omit_centercrop \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${mass_margins_comb_feats_omit_centercrop_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_type_comb_feats_omit_centercrop \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${calc_type_comb_feats_omit_centercrop_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train.py --save \
#        -d calc_dist_comb_feats_omit_centercrop \
#        --njobs 5 \
#        -m 'swin_base_patch4_window12_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 326 \
#        --second_stage_freeze 269 \
#        --best_ckpt_metric 'acc' \
#        -s ${calc_dist_comb_feats_omit_centercrop_save_root}/swin-base-patch4-window12-384_b32_e100_bestacc_384x384_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# cait_m48_448
python train.py --save \
       -d mass_shape_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'cait_m48_448'\
       -b 32 \
       -e 100 -i 448 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 905 \
       --second_stage_freeze 849 \
       --third_stage_freeze 633 \
       -s ${mass_shape_comb_feats_omit_centercrop_save_root}/cait-m48-448_b32_e100_448x448_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d mass_margins_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'cait_m48_448'\
       -b 32 \
       -e 100 -i 448 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 905 \
       --second_stage_freeze 849 \
       --third_stage_freeze 633 \
       -s ${mass_margins_comb_feats_omit_centercrop_save_root}/cait-m48-448_b32_e100_448x448_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_type_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'cait_m48_448'\
       -b 32 \
       -e 100 -i 448 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 905 \
       --second_stage_freeze 849 \
       --third_stage_freeze 633 \
       -s ${calc_type_comb_feats_omit_centercrop_save_root}/cait-m48-448_b32_e100_448x448_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_dist_shape_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'cait_m48_448'\
       -b 32 \
       -e 100 -i 448 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 905 \
       --second_stage_freeze 849 \
       --third_stage_freeze 633 \
       -s ${calc_dist_comb_feats_omit_centercrop_save_root}/cait-m48-448_b32_e100_448x448_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# xcit_small_24_p8_224
python train.py --save \
       -d mass_shape_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p8_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 625 \
       --second_stage_freeze 563 \
       -s ${mass_shape_comb_feats_omit_centercrop_save_root}/xcit-small-24-p8-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d mass_margins_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p8_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 625 \
       --second_stage_freeze 563 \
       -s ${mass_margins_comb_feats_omit_centercrop_save_root}/xcit-small-24-p8-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_type_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p8_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 625 \
       --second_stage_freeze 563 \
       -s ${calc_type_comb_feats_omit_centercrop_save_root}/xcit-small-24-p8-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_dist_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p8_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 625 \
       --second_stage_freeze 563 \
       -s ${calc_dist_comb_feats_omit_centercrop_save_root}/xcit-small-24-p8-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# xcit_small_24_p16_224
python train.py --save \
       -d mass_shape_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p16_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 628 \
       --second_stage_freeze 566 \
       -s ${mass_shape_comb_feats_omit_centercrop_save_root}/xcit-small-24-p16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d mass_margins_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p16_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 628 \
       --second_stage_freeze 566 \
       -s ${mass_margins_comb_feats_omit_centercrop_save_root}/xcit-small-24-p16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_type_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p16_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 628 \
       --second_stage_freeze 566 \
       -s ${calc_type_comb_feats_omit_centercrop_save_root}/xcit-small-24-p16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py --save \
       -d calc_dist_comb_feats_omit_centercrop \
       --njobs 5 \
       -m 'xcit_small_24_p16_224'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --first_stage_freeze 628 \
       --second_stage_freeze 566 \
       -s ${calc_dist_comb_feats_omit_centercrop_save_root}/xcit-small-24-p16-224_b32_e100_224x224_adam_bce_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
