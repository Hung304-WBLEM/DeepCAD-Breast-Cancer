module load cudatoolkit/10.1

test_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/test'
mkdir -p ${test_root}

cd ..

python train.py -d mass_margins_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt bce -s ${test_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py -d calc_type_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt bce -s ${test_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"

python train.py -d calc_dist_comb_feats_omit -m resnet50 -b 32 -e 100 -i 224 --wc --ws --opt adam --crt bce -s ${test_root}/r50_b32_e100_224x224_adam_wc_ws_"$(LC_TIME="EN.UTF-8" date)"
