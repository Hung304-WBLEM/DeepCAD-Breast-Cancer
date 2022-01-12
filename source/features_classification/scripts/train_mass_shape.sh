cd ..
export CUDA_VISIBLE_DEVICES=0,1
# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'resnet50'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_s_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 210 \
#        --second_stage_freeze 168 \

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_m_in21ft1k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 449 \
#        --second_stage_freeze 381

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'tf_efficientnetv2_m_in21ft1k'\
#        -b 32 \
#        -e 3 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 449 \
#        --second_stage_freeze 381

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'xcit_small_24_p16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 628 \
#        --second_stage_freeze 566 \

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'vit_base_patch16_384'\
#        -b 32 \
#        -e 100 -i 384 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'vit_base_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'vit_large_patch16_224_in21k'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 293 \
#        --second_stage_freeze 243

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'dino_vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_mass_shape/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --njobs 5 \
#        -m 'dino_vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape_tfds \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape_tfds \
#        --njobs 5 \
#        -m 'dino_vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --first_stage_freeze 149 \
#        --second_stage_freeze 99

# python run.py \
#        --exp_name mass_shape_full \
#        -d mass_shape\
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name mass_shape_full \
#        -d mass_shape\
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_base_patch16_224'\
#        --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_base_dim8192_five_classes_mass_calc_pathology/checkpoint.pth' \
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

# python run.py \
#        --exp_name mass_shape \
#        -d mass_shape \
#        --one_stage_training \
#        --njobs 5 \
#        -m 'vit_tiny_patch16_224'\
#        -b 32 \
#        -e 100 -i 224 --opt adam --wc --ws --crt bce\
#        --use_lr_scheduler \
#        --first_stage_freeze -1 \
#        --first_stage_lr 0.0001 \
#        --best_ckpt_metric macro_auc

python run.py \
       --exp_name mass_shape \
       -d mass_shape \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_tiny_patch16_224'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_five_classes_mass_calc_pathology/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc


python run.py \
       --exp_name mass_shape \
       -d mass_shape \
       --one_stage_training \
       --njobs 5 \
       -m 'dino_vit_tiny_patch16_224'\
       --ckpt '/home/hqvo2/Projects/Breast_Cancer/libs/dino/experiments/vit_tiny_dim8192_combined_datasets/checkpoint0500.pth' \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc
