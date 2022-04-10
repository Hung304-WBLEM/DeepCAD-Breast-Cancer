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
       --exp_name pathology_4class_simmim \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'simmim_swin_base_maskpatch32_patch16' \
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/SimMIM/outdir/simmim_pretrain/new_simmim_pretrain__swin_base__maskpatch8_075__img192_window6__500ep/ckpt_epoch_499.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_simmim \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'simmim_swin_base_maskpatch32_patch16' \
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/SimMIM/outdir/simmim_pretrain/simmim_pretrain__swin_base__maskpatch4_075__img192_window6__500ep/ckpt_epoch_499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc
