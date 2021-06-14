module load cudatoolkit/10.1

cd ..

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'

mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
mass_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion'
mass_breast_density_image_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_image'

calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'
calc_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion'
calc_breast_density_image_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_image'

# python test.py -s ${four_classes_mass_calc_pathology_save_root}/"r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 15:52:21 CDT 2021"


# python test.py -s ${mass_shape_comb_feats_omit_save_root}/"r50_b32_e100_224x224_adam_bce_wc_ws_Mon Apr  5 06:21:19 CDT 2021"
# python test.py -s ${mass_margins_comb_feats_omit_save_root}/"r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 04:42:39 CDT 2021"
python test.py -s ${mass_breast_density_image_save_root}/"r50_b32_e100_224x224_adam_wc_ws_Sun May 23 09:37:33 CDT 2021"

python test.py -s ${calc_type_comb_feats_omit_save_root}/"r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:10:46 CDT 2021"
python test.py -s ${calc_dist_comb_feats_omit_save_root}/"r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:52:09 CDT 2021"
# python test.py -s ${calc_breast_density_image_save_root}/"r50_b32_e100_224x224_adam_wc_ws_Sun May 23 12:15:57 CDT 2021"
