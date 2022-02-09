cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3


# python run.py \
#        --exp_name mass_margins_mae \
#        -d mass_margins \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_aug_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc_only

# python run.py \
#        --exp_name mass_margins_mae \
#        -d mass_margins \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input112_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc_only
 
# python run.py \
#        --exp_name mass_margins_mae \
#        -d mass_margins \
#        --one_stage_training \
#        --njobs 10 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_vit_base_patch16_input112_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 112 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc_only

python run.py \
       --exp_name mass_margins_mae \
       -d mass_margins \
       --one_stage_training \
       --njobs 10 \
       -m 'mae_vit_base_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_vit_base_patch16_input112_combined_datasets/checkpoint-499.pth' \
       -b 32 \
       -e 100 -i 112 --opt adam --wc --ws --crt bce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc_only
