cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3


python run.py \
       --exp_name calc_type \
       -d calc_type \
       --one_stage_training \
       --njobs 10 \
       -m 'vit_base_patch16_224'\
       --use_pretrained \
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt bce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.00001 \
       --best_ckpt_metric macro_auc_only


