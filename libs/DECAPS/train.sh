#!/bin/bash

#SBATCH -J train_classifier
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=10
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/libs/DECAPS

python train.py --dn cbis_ddsm
