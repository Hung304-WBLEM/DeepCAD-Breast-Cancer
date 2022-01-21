cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_mask0.5_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input64_mask0.5_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e1500_input64_five_classes_mass_calc_pathology/checkpoint-1499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e1500_input64_combined_datasets/checkpoint-1499.pth' \
#        -b 32 \
#        -e 100 -i 64 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-499.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_five_classes_mass_calc_pathology/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.001 \
#        --best_ckpt_metric macro_auc
 
# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16_linprobe'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.001 \
#        --best_ckpt_metric macro_auc


# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-300.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class_mae \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'mae_vit_base_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-450.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.00001 \
#        --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_mae \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'mae_vit_base_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets/checkpoint-499.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc
