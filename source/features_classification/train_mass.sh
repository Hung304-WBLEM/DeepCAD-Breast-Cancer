#!/bin/bash

#SBATCH -J train_mass
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=5
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/source/features_classification


###################################### MASS ################################################
cbis_ddsm_mass_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass_features_classification'


# python train.py -d mass_pathology -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_pathology_r50_b32_e100_adam_wc
# python train.py -d mass_shape_comb_feats_omit -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_shape_comb_feats_r50_b32_e100_adam_wc
# python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_margins_comb_feats_r50_b32_e100_adam_wc
# python train.py -d mass_breast_density_lesion -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_breast_density_lesion_r50_b32_e100_adam_wc

python train.py -d mass_pathology_clean -m resnet50 -b 32 -e 100 -opt adam -wc -s ${cbis_ddsm_mass_features_root}/mass_pathology_clean_r50_b32_e100_adam_wc
