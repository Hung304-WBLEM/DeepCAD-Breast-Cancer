cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3
# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'resnet50'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.003 \
#        --first_stage_wd 0.3
#        --best_ckpt_metric macro_auc
# 
# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 300 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.003 \
#        --first_stage_wd 0.3
#        --best_ckpt_metric macro_auc
# 
# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 300 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --first_stage_wd 0 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
 

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_full \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_full \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_birads34_4class_full \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_birads34_4class_full \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_tiny_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_five_classes_mass_calc_pathology/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_combined_datasets/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_birads34 \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_tiny_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_birads34 \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_five_classes_mass_calc_pathology/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_birads34 \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_combined_datasets/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_birads34 \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_small_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_small_dim65536_combined_datasets/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_birads34 \
#        -d four_classes_mass_calc_pathology_birads34 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_small_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_small_dim65536_combined_datasets_pretrained/checkpoint0500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_small_patch16_224'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_small_dim65536_combined_datasets/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_small_patch16_224'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_small_dim65536_combined_datasets_pretrained/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc
