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

# Test with ground-truth clinical features
# python test_with_additional_features.py \
#        -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_b32_e100_224x224_adam_wc_ws_Mon May  3 06:08:38 CDT 2021'
python test_with_additional_features.py \
       -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_coatt_b32_e100_224x224_adam_wc_ws_Fri Apr 30 14:34:53 CDT 2021'
python test_with_additional_features.py \
       -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_crossatt_b32_e100_224x224_adam_wc_ws_Fri Apr 30 15:29:34 CDT 2021'

# Test with predicted clinical features
# python test_with_additional_features.py \
#        -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_b32_e100_224x224_adam_wc_ws_Mon May  3 06:08:38 CDT 2021' \
#        --use_predicted_feats \
#        --pred_mass_shape ${mass_shape_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Mon Apr  5 06:21:19 CDT 2021/test/test_preds.csv' \
#        --pred_mass_margins ${mass_margins_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 04:42:39 CDT 2021/test/test_preds.csv' \
#        --pred_mass_density_image ${mass_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 09:37:33 CDT 2021/test/test_preds.csv' \
#        --pred_calc_type ${calc_type_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:10:46 CDT 2021/test/test_preds.csv' \
#        --pred_calc_dist ${calc_dist_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:52:09 CDT 2021/test/test_preds.csv' \
#        --pred_calc_density_image ${calc_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 12:15:57 CDT 2021/test/test_preds.csv'


python test_with_additional_features.py \
       -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_coatt_b32_e100_224x224_adam_wc_ws_Fri Apr 30 14:34:53 CDT 2021' \
       --use_predicted_feats \
       --pred_mass_shape ${mass_shape_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Mon Apr  5 06:21:19 CDT 2021/test/test_preds.csv' \
       --pred_mass_margins ${mass_margins_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 04:42:39 CDT 2021/test/test_preds.csv' \
       --pred_mass_density_image ${mass_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 09:37:33 CDT 2021/test/test_preds.csv' \
       --pred_calc_type ${calc_type_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:10:46 CDT 2021/test/test_preds.csv' \
       --pred_calc_dist ${calc_dist_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:52:09 CDT 2021/test/test_preds.csv' \
       --pred_calc_density_image ${calc_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 12:15:57 CDT 2021/test/test_preds.csv'


python test_with_additional_features.py \
       -s '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/with_addtional_features_r50_crossatt_b32_e100_224x224_adam_wc_ws_Fri Apr 30 15:29:34 CDT 2021' \
       --use_predicted_feats \
       --pred_mass_shape ${mass_shape_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Mon Apr  5 06:21:19 CDT 2021/test/test_preds.csv' \
       --pred_mass_margins ${mass_margins_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 04:42:39 CDT 2021/test/test_preds.csv' \
       --pred_mass_density_image ${mass_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 09:37:33 CDT 2021/test/test_preds.csv' \
       --pred_calc_type ${calc_type_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:10:46 CDT 2021/test/test_preds.csv' \
       --pred_calc_dist ${calc_dist_comb_feats_omit_save_root}/'r50_b32_e100_224x224_adam_bce_wc_ws_Tue Apr  6 05:52:09 CDT 2021/test/test_preds.csv' \
       --pred_calc_density_image ${calc_breast_density_image_save_root}/'r50_b32_e100_224x224_adam_wc_ws_Sun May 23 12:15:57 CDT 2021/test/test_preds.csv'

