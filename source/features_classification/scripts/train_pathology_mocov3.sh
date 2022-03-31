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

python run.py \
       --exp_name pathology_4class_mocov3 \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'mocov3_vit_base_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/moco-v3/job_dir/checkpoint_0500.pth.tar' \
       -b 32 \
       -e 100 -i 112 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_mocov3 \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'mocov3_vit_base_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/moco-v3/job_dir/checkpoint_0500.pth.tar' \
       -b 32 \
       -e 100 -i 112 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc
