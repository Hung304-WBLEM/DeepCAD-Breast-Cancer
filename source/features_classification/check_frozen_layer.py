import timm

# from efficientnet_pytorch import EfficientNet
from torchvision import models


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

# EfficientNet V2
all_efficientnet_models = timm.list_models('*efficientnet*ns*', pretrained=True)
model_effnet_v2 = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True)
parameters = model_effnet_v2.named_parameters()
for idx, (name, param) in enumerate(parameters):
    print(idx, name)
print(all_efficientnet_models)

# small
first_stage = 449
second_stage = 381

# medium
first_stage = 646
second_stage = 578

# EfficientNet Noisy Student
# tf_efficientnet_l2_ns_475
model_effnet_ns = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
parameters = model_effnet_ns.named_parameters()
for idx, (name, para) in enumerate(parameters):
    print(idx, name)

# b0
first_stage = 210
second_stage = 142

# b4
first_stage = 415
second_stage = 360
    
# l2_475
first_stage = 1131
second_stage = 1050


vit_model = timm.create_model('resnetv2_101x1_bitm', pretrained=True)
for idx, (name, para) in enumerate(vit_model.named_parameters()):
    print(idx, name)
all_vit_models = timm.list_models('*convit*', pretrained=True)
print(all_vit_models)

# vit_base_patch16_224
first_stage = 149
second_stage = 99

# vit_base_patch16_384
first_stage = 149
second_stage = 99

# vit_base_patch16_224_in21k
first_stage = 149
second_stage = 99

# vit_huge_patch14_224_in21k
first_stage = 391
second_stage = 339

# vit_large_patch16_224_in21k
first_stage = 293
second_stage = 243

# convit_base (torch1.7)
first_stage = 177
second_stage = 123

# twins_svt_base
first_stage = 381
second_stage = 327

# resnetv2_101x1_bitm
first_stage = 303
second_stage = 246
