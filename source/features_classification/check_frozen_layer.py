import timm
import sys
import importlib
import torch

# from efficientnet_pytorch import EfficientNet
from torchvision import models
# from utils import load_for_transfer_learning
t2t_vit = importlib.import_module('T2T-ViT.models.t2t_vit')
t2t_vit_utils = importlib.import_module('T2T-ViT.utils')


# # ResNet
# model_resnet = models.resnet50(pretrained=False)
# # print(len(list(model_resnet.named_parameters())))

# # EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b7')
# print(model._fc.in_features)
# parameters = model.named_parameters()
# for idx, (name, param) in enumerate(parameters):
#     print(idx, name)
# # print(len(list(parameters)))
# # print(model)

# # b0
# first_stage = 210
# second_stage = 168

# # b4
# first_stage = 415
# second_stage = 360

# # b7
# first_stage = 708
# second_stage = 653

# # EfficientNet V2
# all_efficientnet_models = timm.list_models('*efficientnet*ns*', pretrained=True)
# model_effnet_v2 = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True)
# parameters = model_effnet_v2.named_parameters()
# for idx, (name, param) in enumerate(parameters):
#     print(idx, name)
# print(all_efficientnet_models)

# # small
# first_stage = 449
# second_stage = 381

# # medium
# first_stage = 646
# second_stage = 578

# # EfficientNet Noisy Student
# # tf_efficientnet_l2_ns_475
# model_effnet_ns = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
# parameters = model_effnet_ns.named_parameters()
# for idx, (name, para) in enumerate(parameters):
#     print(idx, name)

# # b0
# first_stage = 210
# second_stage = 142

# # b4
# first_stage = 415
# second_stage = 360
    
# # l2_475
# first_stage = 1131
# second_stage = 1050


# vit_model = timm.create_model('resnetv2_101x1_bitm', pretrained=True)
# for idx, (name, para) in enumerate(vit_model.named_parameters()):
#     print(idx, name)
# all_vit_models = timm.list_models('*convit*', pretrained=True)
# print(all_vit_models)

# # vit_base_patch16_224
# first_stage = 149
# second_stage = 99

# # vit_base_patch16_384
# first_stage = 149
# second_stage = 99

# # vit_base_patch16_224_in21k
# first_stage = 149
# second_stage = 99

# # vit_huge_patch14_224_in21k
# first_stage = 391
# second_stage = 339

# # vit_large_patch16_224_in21k
# first_stage = 293
# second_stage = 243

# # convit_base (torch1.7)
# first_stage = 177
# second_stage = 123

# # twins_svt_base
# first_stage = 381
# second_stage = 327

# # resnetv2_101x1_bitm
# first_stage = 303
# second_stage = 246

# T2T-ViT
# create model
# model = t2t_vit.t2t_vit_14(num_classes=3)
# model = t2t_vit.t2t_vit_19(num_classes=3)
# model = t2t_vit.t2t_vit_14(num_classes=3)
# model = t2t_vit.t2t_vit_t_14(num_classes=3)
# model = t2t_vit.t2t_vit_t_19(num_classes=3)

# # load the pretrained weights
# t2t_vit_utils.load_for_transfer_learning(model,
#                                          # '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.5_T2T_ViT_14.pth.tar',
#                                          # '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.9_T2T_ViT_19.pth.tar',
#                                          # '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/83.3_T2T_ViT_14.pth.tar',
#                                          # '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.7_T2T_ViTt_14.pth.tar',
#                                          '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/82.4_T2T_ViTt_19.pth.tar',
#                                          use_ema=True,
#                                          strict=False,
#                                          num_classes=3)  # change num_classes based on dataset, can work for different image size as we interpolate the position embeding for different image size.


# for idx, (name, param) in enumerate(model.named_parameters()):
#     print(idx, name)

# # T2T-ViT-14 and T2T-ViT-14_384
# first_stage = 185
# second_stage = 128
              
# # T2T-ViT-19
# first_stage = 240
# second_stage = 194

# # T2T-ViT_t-14
# first_stage = 181
# second_stage = 124

# # T2T-ViT_t-19
# first_stage = 236
# second_stage = 190

# CaiT, CoaT, DeiT, PiT, TNT, Swin, XCiT
model = timm.create_model('cait_m48_448', pretrained=True)
for idx, (name, para) in enumerate(model.named_parameters()):
    print(idx, name)

all_cait_models = timm.list_models('*xcit_small*', pretrained=True)
print(all_cait_models)

# cait_m36_384
first_stage = 689
second_stage = 633
third_stage = 300

# cait_s36_384
first_stage = 689
second_stage = 633

# coat_lite_small
first_stage = 245
second_stage = 195

# deit_base_patch16_224
first_stage = 149
second_stage = 99

# deit_base_patch16_384
first_stage = 149
second_stage = 99

# pit_b_224
first_stage = 169
second_stage = 119

# pit_b_distilled_224
first_stage = 171
second_stage = 119

# tnt_s_patch16_224
first_stage = 348
second_stage = 290

# swin_base_patch4_window7_224
first_stage = 326
second_stage = 269

# swin_base_patch4_window12_384
first_stage = 326
second_stage = 269

# crossvit_18_240
first_stage = 371
second_stage = 319

# #######
# cait_m48_448
first_stage = 905
second_stage = 849
third_stage = 543

# xcit_small_24_p8_224
first_stage = 625
second_stage = 563

# xcit_small_24_p16_224
first_stage = 628
second_stage = 566
