module load cudatoolkit/10.1

four_classes_mass_calc_pathology_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology'

mkdir -p ${four_classes_mass_calc_pathology_save_root}

cd ..


# Four classes Mass Calcification Pathology
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_r50_att_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Emb Concat
python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --missing_feats_fill emp_sampling -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_emp_sampling_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Emb Concat with uncertainty
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.5 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.4 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.3 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.2 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.1 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --train_uct 0.3 --test_uc 0.3 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Co-Attention
python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --missing_feats_fill emp_sampling -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_emp_sampling_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Co-Attention with uncertainty
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.5 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.4 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.3 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.2 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.1 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --train_uct 0.3 --test_uc 0.3 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_coatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Cross-Attention
python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --missing_feats_fill emp_sampling -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_emp_sampling_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Cross-Attention with uncertainty
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.5 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.4 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.3 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.2 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.1 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0 --test_uc 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --train_uct 0.3 --test_uc 0.3 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_uncertainty_r50_crossatt_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Emb Concat with missed clinical features
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --missed_feats_num 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse concat --missed_feats_num 2 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Co-Attention with missed clinical features
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --missed_feats_num 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse coatt --missed_feats_num 2 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

# Cross-Attention with missed clinical features
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --missed_feats_num 1 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
# python train_with_additional_features.py --njobs 8 -d four_classes_mass_calc_pathology -m resnet50 -b 32 -e 100 -i 224 --opt adam --wc --ws --fuse crossatt --missed_feats_num 2 -s ${four_classes_mass_calc_pathology_save_root}/with_addtional_features_missedfeats_r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
