#!/bin/bash

#SBATCH -J br_density_img
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=10
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/source/features_classification


###################################### MASS ################################################
cbis_ddsm_mass_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass_features_classification'
python train.py -d mass_breast_density_image -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_breast_density_image_r50_b32_e100_adam_wc


##################################### CALC #####################################3

cbis_ddsm_calc_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_calc_features_classification'
python train.py -d calc_breast_density_image -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_calc_features_root}/calc_breast_density_image_r50_b32_e100_adam_wc
