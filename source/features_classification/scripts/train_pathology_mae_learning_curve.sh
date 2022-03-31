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
       

for i in `seq 0.1 0.1 1`
do
    python run.py \
           --exp_name pathology_4class_mae_plotting_curve \
           -d four_classes_mass_calc_pathology \
           --train_rate $i \
           --one_stage_training \
           --njobs 10 \
           -m 'mae_vit_base_patch16'\
           --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v2_nomask_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
           -b 32 \
           -e 100 -i 112 --opt adam --wc --ws --crt ce\
           --use_lr_scheduler \
           --first_stage_freeze -1 \
           --first_stage_lr 0.00001 \
           --best_ckpt_metric macro_auc
done
