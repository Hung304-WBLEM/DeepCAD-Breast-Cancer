#!/bin/bash

#SBATCH -J train_classifier
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=4
#SBATCH -t 20:00:00
#SBATCH --mem-per-cpu=4096

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1


mass_calc_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_features_classification'

python train.py --dn four_classes_mass_calc_pathology -e 100 -b 32 --ih 224 --iw 224 --nc 4 --sd ${mass_calc_save_root}/four_classes_mass_calc_pathology_capsnet_e100_b32_224x224
