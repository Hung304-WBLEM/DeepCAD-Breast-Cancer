cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3


python run.py \
       --exp_name pathology_4class_clinical_only \
       -d four_classes_features_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'clinical_default'\
       --use_clinical_feats_only \
       -b 32 \
       -e 100 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.01 \
       --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_clinical_only \
       -d four_classes_features_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'clinical_default'\
       --use_clinical_feats_only \
       -b 32 \
       -e 100 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.001 \
       --best_ckpt_metric macro_auc

python run.py \
       --exp_name pathology_4class_clinical_only \
       -d four_classes_features_pathology \
       --one_stage_training \
       --njobs 10 \
       -m 'clinical_default'\
       --use_clinical_feats_only \
       -b 32 \
       -e 100 --opt adam --wc --ws --crt ce\
       --use_lr_scheduler \
       --first_stage_freeze -1 \
       --first_stage_lr 0.0001 \
       --best_ckpt_metric macro_auc
