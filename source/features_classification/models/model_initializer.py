import timm
import torch.nn as nn

from torchvision import datasets, models, transforms
from features_classification.models.dino_pretrained import ViT_DINO

def set_parameter_requires_grad(model, model_name, last_frozen_layer):
    '''
    Parameters:
    model_name - can be 'vgg16' or 'resnet50'
    freeze_type - can be 'none', 'all'
    '''
    for idx, (name, param) in enumerate(model.named_parameters()):
        param.requires_grad = True

    for idx, (name, param) in enumerate(model.named_parameters()):
        print(idx, name)

        if idx <= last_frozen_layer:
            param.requires_grad = False


def initialize_model(model_name, num_classes, use_pretrained=True, ckpt_path=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == "dilated_resnet50":
        """ Dilated Resnet50
        """
        model_ft = \
            models.resnet50(pretrained=use_pretrained,
                            replace_stride_with_dilation=[options.resnet_dilated_layer2,
                                                          options.resnet_dilated_layer3,
                                                          options.resnet_dilated_layer4])
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    elif 'efficientnet' in model_name:

        if 'tf_efficientnet' in model_name:
            model_ft = timm.create_model(model_name,
                                         pretrained=True, num_classes=num_classes)
        else:
            model_ft = EfficientNet.from_pretrained(model_name)
            num_ftrs = model_ft._fc.in_features
            model_ft._fc = nn.Linear(num_ftrs, num_classes)

    elif 'T2T-ViT' in model_name:
        if model_name == 'T2T-ViT-14':
            model_ft = t2t_vit.t2t_vit_14(num_classes=num_classes)
            t2t_vit_utils.load_for_transfer_learning(model_ft,
                                                     '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.5_T2T_ViT_14.pth.tar',
                                                     use_ema=True,
                                                     strict=False,
                                                     num_classes=num_classes)  
        elif model_name == 'T2T-ViT-19':
            model_ft = t2t_vit.t2t_vit_19(num_classes=num_classes)
            t2t_vit_utils.load_for_transfer_learning(model_ft,
                                                     '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.9_T2T_ViT_19.pth.tar',
                                                     use_ema=True,
                                                     strict=False,
                                                     num_classes=num_classes)  
        elif model_name == 'T2T-ViT-14_384':
            model_ft = t2t_vit.t2t_vit_14(img_size=384,
                                          num_classes=num_classes)
            t2t_vit_utils.load_for_transfer_learning(model_ft,
                                                     '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/83.3_T2T_ViT_14.pth.tar',
                                                     use_ema=True,
                                                     strict=False,
                                                     num_classes=num_classes)  
        elif model_name == 'T2T-ViT_t-14':
            model_ft = t2t_vit.t2t_vit_t_14(num_classes=num_classes)
            t2t_vit_utils.load_for_transfer_learning(model_ft,
                                                     '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/81.7_T2T_ViTt_14.pth.tar',
                                                     use_ema=True,
                                                     strict=False,
                                                     num_classes=num_classes)  
        elif model_name == 'T2T-ViT_t-19':
            model_ft = t2t_vit.t2t_vit_t_19(num_classes=num_classes)
            t2t_vit_utils.load_for_transfer_learning(model_ft,
                                                     '/home/hqvo2/Projects/Breast_Cancer/libs/T2T-ViT/ckpts/82.4_T2T_ViTt_19.pth.tar',
                                                     use_ema=True,
                                                     strict=False,
                                                     num_classes=num_classes)  

        
    elif ('dino' not in model_name) and \
         ('vit' in model_name \
         or 'twins' in model_name \
         or 'bit' in model_name \
         or 'T2T-ViT' in model_name \
         or 'cait' in model_name \
         or 'coat' in model_name \
         or 'deit' in model_name \
         or 'pit' in model_name \
         or 'tnt' in model_name \
         or 'swin' in model_name \
         or 'xcit' in model_name):

        model_ft = timm.create_model(model_name,
                                     pretrained=use_pretrained,
                                     num_classes=num_classes)

    elif 'dino' in model_name:
        if model_name == 'dino_vit_base_patch16_224':
            model_ft = ViT_DINO(ckpt_path, 'vit_base', 16, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':

        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft