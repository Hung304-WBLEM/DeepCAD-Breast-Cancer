mass_shape_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit'
mass_margins_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_margins_comb_feats_omit'
calc_type_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_type_comb_feats_omit'
calc_dist_comb_feats_omit_save_root='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/calc_dist_comb_feats_omit'

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --num_steps 10000 --fp16 --name input224 --data_root . --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' --pretrained_model '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit/transfg_b16_e100_224x224/ckpt.bin'

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins --img_size 224 --split overlap --num_steps 1200 --fp16 --name mass_margins_224x224 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin \
# 		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins --img_size 224 --split overlap --fp16 --name mass_margins_224x224_e5000 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e5000_224x224/ckpt.bin \
# 		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e5000_224x224

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Type --img_size 224 --split overlap --fp16 --name calc_type_224x224 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin \
# 		    --save_path ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Dist --img_size 224 --split overlap --fp16 --name calc_dist_224x224 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224/ckpt.bin \
# 		    --save_path ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --fp16 --name mass_shape_224x224_slidestep-10 --slide_step 10 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-10/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-10

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --fp16 --name mass_shape_224x224_slidestep-8 --slide_step 8 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-8/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-8

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --fp16 --name mass_shape_224x224_slidestep-6 --slide_step 6 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-6/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_stepsize-6

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --criterion bce --wc --fp16 --name mass_shape_224x224_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins --img_size 224 --split overlap --criterion bce --wc --fp16 --name mass_margins_224x224_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin \
# 		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Type --img_size 224 --split overlap --criterion bce --wc --fp16 --name calc_type_224x224_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin \
# 		    --save_path ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Dist --img_size 224 --split overlap --criterion bce --wc --fp16 --name calc_dist_224x224_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc/ckpt.bin \
# 		    --save_path ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape  --split overlap --img_size 448 --criterion bce --wc --fp16 --name mass_shape_448x448_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins  --split overlap --criterion bce --wc --fp16 --name mass_margins_448x448_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin \
# 		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Type  --split overlap --criterion bce --wc --fp16 --name calc_type_448x448_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin \
# 		    --save_path ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Dist  --split overlap --criterion bce --wc --fp16 --name calc_dist_448x448_bce_wc --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc/ckpt.bin \
# 		    --save_path ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_448x448_bce_wc

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --fp16 --name mass_shape_224x224_b16_bce_wc_slidestep-6 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin \
# 		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --fp16 --name mass_margins_224x224_b16_bce_wc_slidestep-6 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin \
# 		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Type --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --fp16 --name calc_type_224x224_b16_bce_wc_slidestep-6 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin \
# 		    --save_path ${calc_type_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Dist --img_size 224 --split overlap --slide_step 6 --criterion bce --wc --fp16 --name calc_dist_224x224_b16_bce_wc_slidestep-6 --data_root . \
# 		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
# 		    --pretrained_model ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6/ckpt.bin \
# 		    --save_path ${calc_dist_comb_feats_omit_save_root}/transfg_b16_e100_224x224_b16_bce_wc_slidestep-6

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Shape --img_size 224 --split overlap --criterion bce --wc --fp16 --name mass_shape_224x224_bce_wc --data_root . \
		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
		    --pretrained_model ${mass_shape_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin \
		    --save_path ${mass_shape_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Mass_Margins --img_size 224 --split overlap --criterion bce --wc --fp16 --name mass_margins_224x224_bce_wc --data_root . \
		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
		    --pretrained_model ${mass_margins_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin \
		    --save_path ${mass_margins_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Type --img_size 224 --split overlap --criterion bce --wc --fp16 --name calc_type_224x224_bce_wc --data_root . \
		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
		    --pretrained_model ${calc_type_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin \
		    --save_path ${calc_type_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 test_transfg.py --dataset Calc_Dist --img_size 224 --split overlap --criterion bce --wc --fp16 --name calc_dist_224x224_bce_wc --data_root . \
		    --pretrained_dir '/home/hqvo2/Projects/Breast_Cancer/libs/TransFG/ViT-B_16.npz' \
		    --pretrained_model ${calc_dist_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc/ckpt.bin \
		    --save_path ${calc_dist_comb_feats_omit_save_root}/transfg_b32_e100_224x224_bce_wc
