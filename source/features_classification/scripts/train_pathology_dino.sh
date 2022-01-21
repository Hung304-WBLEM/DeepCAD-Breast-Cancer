cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3


# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim8192_imgsize56_five_classes_mass_calc_pathology/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim8192_imgsize56_combined_datasets/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim4096_imgsize56_five_classes_mass_calc_pathology/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim4096_imgsize56_combined_datasets/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim2048_imgsize56_five_classes_mass_calc_pathology/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim2048_imgsize56_combined_datasets/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim1024_imgsize56_five_classes_mass_calc_pathology/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim1024_imgsize56_combined_datasets/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim512_imgsize56_five_classes_mass_calc_pathology/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name pathology_4class \
#        -d four_classes_mass_calc_pathology \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'dino_vit_tiny_patch16'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim512_imgsize56_combined_datasets/checkpoint0450.pth' \
#        -b 32 \
#        -e 100 -i 56 --opt adam --wc --ws --crt ce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_dino \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_tiny_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim2048_imgsize64_five_classes_mass_calc_pathology/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 64 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_dino \
       -d four_classes_mass_calc_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_tiny_patch16'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_outdim2048_imgsize64_combined_datasets/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 64 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc
