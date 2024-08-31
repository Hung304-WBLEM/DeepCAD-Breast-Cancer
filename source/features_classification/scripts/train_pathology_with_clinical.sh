cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3

python run.py \
       --exp_name pathology_4class_with_clinical_feats \
       -d four_classes_features_pathology \
       --one_stage_training \
       --njobs 5 \
       -m 'fusion_resnet50'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc \
       --use_clinical_feats

python run.py \
       --exp_name pathology_4class_with_clinical_feats \
       -d four_classes_features_pathology \
       --use_pretrained \
       --one_stage_training \
       --njobs 10 \
       -m 'fusion_parallel_resnet50'\
       -b 32 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce_rank\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc \
       --use_clinical_feats

python run.py \
       --exp_name pathology_4class_with_clinical_feats \
       -d four_classes_features_pathology \
       --use_pretrained \
       --one_stage_training \
       --njobs 10 \
       -m 'fusion_parallel_resnet50'\
       -b 64 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce_rank\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc \
       --use_clinical_feats

python run.py \
       --exp_name pathology_4class_with_clinical_feats \
       -d four_classes_features_pathology \
       --use_pretrained \
       --one_stage_training \
       --njobs 10 \
       -m 'fusion_parallel_resnet50'\
       -b 128 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce_rank\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc \
       --use_clinical_feats

python run.py \
       --exp_name pathology_4class_with_clinical_feats \
       -d four_classes_features_pathology \
       --use_pretrained \
       --one_stage_training \
       --njobs 10 \
       -m 'fusion_parallel_resnet50'\
       -b 256 \
       -e 100 -i 224 --opt adam --wc --ws --crt ce_rank\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc \
       --use_clinical_feats

for batchsize in 32 64 128 256
do
    python run.py \
           --exp_name pathology_4class_with_clinical_feats \
           -d four_classes_features_pathology \
           --use_pretrained \
           --one_stage_training \
           --njobs 10 \
           -m 'fusion_parallel_resnet50'\
           --sim_func dot \
           --margin 3 \
           -b $batchsize \
           -e 100 -i 224 --opt adam --wc --ws --crt ce_rank\
           --use_lr_scheduler \
           --first_stage_freeze -1 \
           --first_stage_lr 0.0001 \
           --best_ckpt_metric macro_auc \
           --use_clinical_feats
done
