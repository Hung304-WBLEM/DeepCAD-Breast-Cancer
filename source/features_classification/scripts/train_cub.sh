cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3


# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_tiny_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'resnet50'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_with_clinical_feats \
#        -d four_classes_features_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'fusion_resnet50'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc \
#        --use_clinical_feats

# python run.py \
#        --exp_name pathology_4class \
#        -d cub_200_2011 \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_tiny_patch16_224'\
#        --use_pretrained \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


python run.py \
       --exp_name pathology_4class \
       -d cub_200_2011 \
       --one_stage_training \
       --njobs 5 \
       -m 'vit_base_patch16_224'\
       --use_pretrained \
       -b 32 \
       -e 100 -i 112 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc
