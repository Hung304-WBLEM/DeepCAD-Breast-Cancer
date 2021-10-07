module load cudatoolkit/10.1

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'
four_classes_mass_calc_pathology_histeq_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology_histeq'

mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
mass_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion'

calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'
calc_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion'
calc_breast_density_image_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_image'

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



for i in {5..9}
do
    python train.py --njobs 8 -d calc_breast_density_image -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --tr 0.${i} -s ${calc_breast_density_image_save_root}/r50_b32_e100_224x224_adam_wc_ws_tr0.${i}_"$(LC_TIME="EN.UTF-8" date)"
done
