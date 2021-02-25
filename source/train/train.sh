#!/bin/bash

#SBATCH -J train_classifiers_without_additional_features
#SBATCH -o result.o%j
#SBATCH -N 1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=10
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=2048

#SBATCH --mail-user=voquochung304@gmail.com
#SBATCH --mail-type=all

module load cudatoolkit/10.1

cd /home/hqvo2/Projects/Breast_Cancer/source/train


###################################### MASS ################################################
cbis_ddsm_mass_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_mass_features_classification'

# train resnet50 for mass pathology
r50_frozen_types=('none' 'all' 'last_fc' 'top1_conv_block' 'top2_conv_block' 'top3_conv_block')
for frozen_type in ${r50_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d mass_pathology -m resnet50 -b 128 -e 100 -f $frozen_type -s ${cbis_ddsm_mass_features_root}/mass_pathology_r50_frozen-${frozen_type}_b32_e100
done

# train vgg16 for mass pathology
vgg16_frozen_types=('none' 'all' 'last_fc' 'fc2' 'fc1' 'top1_conv_block' 'top2_conv_block')
for frozen_type in ${vgg16_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d mass_pathology -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_mass_features_root}/mass_pathology_vgg16_frozen-${frozen_type}_b32_e100
done

# train resnet50 for mass features
mass_feats=('mass_shape_comb_feats_omit' 'mass_margins_comb_feats_omit' 'mass_breast_density_lesion')
# mass_feats=('mass_breast_density_image')
r50_frozen_types=('top1_conv_block')
for mass_feat in ${mass_feats[@]}; do
    for frozen_type in ${r50_frozen_types[@]}; do
	echo $frozen_type
	python train.py -d $mass_feat -m resnet50 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_mass_features_root}/${mass_feat}_r50_frozen-${frozen_type}_b32_e100
    done
done

# train vgg16 for mass features
vgg16_frozen_types=('top1_conv_block')
for mass_feat in ${mass_feats[@]}; do
    for frozen_type in ${vgg16_frozen_types[@]}; do
	echo $frozen_type
	python train.py -d $mass_feat -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_mass_features_root}/${mass_feat}_vgg16_frozen-${frozen_type}_b32_e100
    done
done


##################################### CALC #####################################3

cbis_ddsm_calc_features_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_calc_features_classification'

# train resnet50 for calc pathology
r50_frozen_types=('none' 'all' 'last_fc' 'top1_conv_block' 'top2_conv_block' 'top3_conv_block')
r50_frozen_types=('top1_conv_block')
for frozen_type in ${r50_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d calc_pathology -m resnet50 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/calc_pathology_r50_frozen-${frozen_type}_b32_e100
done

# train vgg16 for calc pathology
vgg16_frozen_types=('none' 'all' 'last_fc' 'fc2' 'fc1' 'top1_conv_block' 'top2_conv_block')
for frozen_type in ${vgg16_frozen_types[@]}; do
    echo $frozen_type
    python train.py -d calc_pathology -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/calc_pathology_vgg16_frozen-${frozen_type}_b32_e100
done

# train resnet50 for calc features
calc_feats=('calc_type_comb_feats_omit' 'calc_dist_comb_feats_omit' 'calc_breast_density_lesion')
# calc_feats=('calc_breast_density_image')
r50_frozen_types=('top1_conv_block')

for calc_feat in ${calc_feats[@]}; do
    for frozen_type in ${r50_frozen_types[@]}; do
	echo $frozen_type
	python train.py -d $calc_feat -m resnet50 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/${calc_feat}_r50_frozen-${frozen_type}_b32_e100
    done
done

# train vgg16 for calc features
vgg16_frozen_types=('top1_conv_block')
for calc_feat in ${calc_feats[@]}; do
    for frozen_type in ${vgg16_frozen_types[@]}; do
	echo $frozen_type
	python train.py -d $calc_feat -m vgg16 -b 32 -e 100 -f $frozen_type -s ${cbis_ddsm_calc_features_root}/${calc_feat}_vgg16_frozen-${frozen_type}_b32_e100
    done
done
