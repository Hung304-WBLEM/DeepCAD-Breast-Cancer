#!/bin/bash

#SBATCH -J train_classifiers_without_additional_features
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=10
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/source/train

cbis_ddsm_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification'

# train nets for pathology
# r50_frozen_types=('none' 'all' 'last_fc' 'top1_conv_block' 'top2_conv_block' 'top3_conv_block')
# for frozen_type in ${r50_frozen_types[@]}; do
#     echo $frozen_type
#     python train.py -d mass_pathology -m resnet50 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_features_root}/mass_pathology_r50_frozen-${frozen_type}_b32_e100
# done
# 
# 
# vgg16_frozen_types=('none' 'all' 'last_fc' 'fc2' 'fc1' 'top1_conv_block' 'top2_conv_block')
# for frozen_type in ${vgg16_frozen_types[@]}; do
#     echo $frozen_type
#     python train.py -d mass_pathology -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_features_root}/mass_pathology_vgg16_frozen-${frozen_type}_b32_e100
# done

cbis_ddsm_calc_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_calc_features_classification'
r50_frozen_types=('none' 'all' 'last_fc' 'top1_conv_block' 'top2_conv_block' 'top3_conv_block')
for frozen_type in ${r50_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d calc_pathology -m resnet50 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/calc_pathology_r50_frozen-${frozen_type}_b32_e100
done


vgg16_frozen_types=('none' 'all' 'last_fc' 'fc2' 'fc1' 'top1_conv_block' 'top2_conv_block')
for frozen_type in ${vgg16_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d calc_pathology -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/calc_pathology_vgg16_frozen-${frozen_type}_b32_e100
done
