module load cudatoolkit/10.1

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'

mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
mass_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_breast_density_lesion'

calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'
calc_breast_density_lesion_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_breast_density_lesion'

mkdir -p ${four_classes_mass_calc_pathology_save_root}
mkdir -p ${mass_shape_comb_feats_omit_save_root}
mkdir -p ${mass_margins_comb_feats_omit_save_root}
mkdir -p ${mass_breast_density_lesion_save_root}
mkdir -p ${calc_type_comb_feats_omit_save_root}
mkdir -p ${calc_dist_comb_feats_omit_save_root}
mkdir -p ${calc_breast_density_lesion_save_root}

cd ..

python train.py -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_shape_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_margins_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
python train.py -d mass_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${mass_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py -d calc_type_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_type_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
python train.py -d calc_dist_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_dist_comb_feats_omit_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
python train.py -d calc_breast_density_lesion -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${calc_breast_density_lesion_save_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
