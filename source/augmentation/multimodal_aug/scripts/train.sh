# python train.py --save --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_b64_e20_64x64_adam_"$(LC_TIME="EN.UTF-8" date)"
# python train.py --save \
#        --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_b64_e20_64x64_adam_"$(LC_TIME="EN.UTF-8" date)" \
#        --rl 0.9 \
#        --fl 0.1


# python train.py --save \
#        --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_b64_e20_64x64_adam_random-label-smooth_"$(LC_TIME="EN.UTF-8" date)" \
#        --lsmooth \
#        --rl_min 0.7 \
#        --rl_max 1.2 \
#        --fl_min 0.0 \
#        --fl_max 0.3


# python train.py --save \
#        --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_b64_e20_64x64_adam_wasserstein_d-iters-2_random-label-smooth_"$(LC_TIME="EN.UTF-8" date)" \
#        --lsmooth \
#        --rl_min 0.7 \
#        --rl_max 1.2 \
#        --fl_min 0.0 \
#        --fl_max 0.3 \
#        --loss_func wasserstein \
#        --d_num_iters 2

python train.py --save \
       --ngpu 1 \
       --njobs 8 \
       --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_coatt_b100_e300_32x32_adam_wasserstein_d-iters-1_random-label-smooth_"$(LC_TIME="EN.UTF-8" date)" \
       --lsmooth \
       --rl_min 0.7 \
       --rl_max 1.2 \
       --fl_min 0.0 \
       --fl_max 0.3 \
       --loss_func wasserstein \
       --d_num_iters 5 \
       --d_fuse_type coatt \
       --d_lr 0.0004 \
       --g_lr 0.0002 \
       -b 100 \
       --epochs 300 \
       --ims 32

# python train.py --save \
#        --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_crossatt_b64_e20_64x64_adam_wasserstein_d-iters-2_random-label-smooth_"$(LC_TIME="EN.UTF-8" date)" \
#        --lsmooth \
#        --rl_min 0.7 \
#        --rl_max 1.2 \
#        --fl_min 0.0 \
#        --fl_max 0.3 \
#        --loss_func wasserstein \
#        --d_num_iters 1 \
#        --d_fuse_type crossatt


# python train.py --save \
#        --njobs 8 \
#        --save_path /home/hqvo2/Projects/Breast_Cancer/experiments/mnist_augmentation/r50_coatt_b64_e200_64x64_adam_wasserstein_d-iters-1_random-label-smooth_"$(LC_TIME="EN.UTF-8" date)" \
#        --lsmooth \
#        --rl_min 0.7 \
#        --rl_max 1.2 \
#        --fl_min 0.0 \
#        --fl_max 0.3 \
#        --loss_func wasserstein \
#        --d_num_iters 1 \
#        --d_fuse_type coatt \
#        --epochs 200 \
#        --ims 32

