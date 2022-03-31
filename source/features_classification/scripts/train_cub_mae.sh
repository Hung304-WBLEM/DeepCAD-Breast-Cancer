#!/bin/bash

#SBATCH -J mae
#SBATCH -o result.o%j
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all



####################################################################################################################################33

cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=1,2,3,5

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_mask0.5_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_mask0.5_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e1500_input64_five_classes_mass_calc_pathology/checkpoint-1499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e1500_input64_combined_datasets/checkpoint-1499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-799.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-799.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-799.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_wd0.01_input224_image_lesion_combined_datasets/checkpoint-500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_blr1.5e-3_input224_image_lesion_combined_datasets/checkpoint-500.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc
# 

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-350.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-350.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_aug_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_aug_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.01 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_vit_base_patch16_input224_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input112_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_vit_base_patch16_input112_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ip_2gpu_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ip_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ip_nomask_vit_base_patch16_input112_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_nomask_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input112_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/newtest_vit_base_patch16_e500_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_nomaskboth_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ip_nomaskboth_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ip_lr1.5e-3_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_ms_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_msv2_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_circularmask_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v2_nomask_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v2_ip_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v2_nomaskboth_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v3_linformer_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v3_linformer-k128_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v3_performer_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v3_nystromer_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

python run.py \
       --exp_name cub_mae \
       -d cub_200_2011 \
       --one_stage_training \
       --njobs 10 \
       -m 'mae_vit_base_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input112_cub_200_2011/checkpoint-499.pth' \
       -b 1024 \
       -e 3 -i 112 --opt adam --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc

