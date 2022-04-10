import timm
import torch.nn as nn

from torchvision import datasets, models, transforms
from features_classification.models.dino_pretrained import ViT_DINO
from features_classification.models.mae_pretrained import ViT_MAE
from features_classification.models.mocov3_pretrained import ViT_Mocov3 
from features_classification.models.simmim_pretrained import Swin_SimMIM
from features_classification.models.fusion_models.clinical_feats_models import Clinical_Concat_Model, Clinical_Attentive_Model, Clinical_Parallel_Model
from features_classification.models.clinical_models.clinical_models import Clinical_Model


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


def initialize_model(options, model_name, num_classes, use_pretrained=True, ckpt_path=None):
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

        
    elif ('dino' not in model_name
          and 'mae' not in model_name
          and 'mocov3' not in model_name
          and 'simmim' not in model_name
          ) \
          and \
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

        if model_name == 'vit_tiny_patch8':
            from mae import models_vit

            model_ft = models_vit.__dict__['vit_tiny_patch8'](
                num_classes=num_classes,
                drop_path_rate=0.1,
                global_pool=True,
                img_size=options.input_size
            )
        else:
            model_ft = timm.create_model(model_name,
                                         pretrained=use_pretrained,
                                         num_classes=num_classes)

    elif 'dino' in model_name:
        if model_name in ['dino_vit_tiny_patch16']:
            model_ft = ViT_DINO(ckpt_path, 'vit_tiny', options.input_size, 16, num_classes)
        elif model_name == 'dino_vit_small_patch16':
            model_ft = ViT_DINO(ckpt_path, 'vit_small', options.input_size, 16, num_classes)
        elif model_name == 'dino_vit_base_patch16':
            model_ft = ViT_DINO(ckpt_path, 'vit_base', options.input_size, 16, num_classes)

    elif 'mae' in model_name:
        if 'linprobe' not in model_name:
            if model_name in ['mae_vit_base_patch16']:
                model_ft = ViT_MAE(ckpt_path, 'vit_base_patch16', options.input_size,
                                num_classes, global_pool=True)
            elif model_name in ['mae_vit_large_patch16']:
                model_ft = ViT_MAE(ckpt_path, 'vit_large_patch16', options.input_size,
                                num_classes, global_pool=True)
        elif 'linprobe' in model_name:
            if model_name in ['mae_vit_base_patch16_linprobe']:
                model_ft = ViT_MAE(ckpt_path, 'vit_base_patch16', options.input_size,
                                   num_classes, global_pool=True, linprobe=True)
            elif model_name in ['mae_vit_large_patch16_linprobe']:
                model_ft = ViT_MAE(ckpt_path, 'vit_large_patch16', options.input_size,
                                   num_classes, global_pool=True, linprobe=True)

    elif 'mocov3' in model_name:
        if model_name in ['mocov3_vit_base_patch16']:
            model_ft = ViT_Mocov3(ckpt_path, 'vit_base', options.input_size,
                                num_classes)

    elif 'simmim' in model_name:
        if model_name in ['simmim_swin_base_maskpatch32_patch16']:
            model_ft = Swin_SimMIM(ckpt_path, 'swin_base_patch4_window7',
                                   options.input_size, num_classes)
        
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

    elif 'fusion' in model_name:
        breast_density_cats = 4
        mass_shape_cats= 8
        mass_margins_cats = 5
        calc_type_cats = 14
        calc_dist_cats = 5

        if model_name == 'fusion_resnet50':
            if options.dataset in ['four_classes_features_pathology']:
                if options.fusion_type == 'concat':
                    model_ft = \
                        Clinical_Concat_Model(
                            model_name, 
                            input_vector_dim=\
                            breast_density_cats+\
                            mass_shape_cats+\
                            mass_margins_cats+\
                            calc_type_cats+\
                            calc_dist_cats, num_classes=num_classes)

        elif model_name == 'fusion_parallel_resnet50':
            if options.dataset in ['four_classes_features_pathology']:
                if options.fusion_type == 'concat':
                    model_ft = \
                        Clinical_Parallel_Model(
                            model_name, 
                            input_vector_dim=\
                            breast_density_cats+\
                            mass_shape_cats+\
                            mass_margins_cats+\
                            calc_type_cats+\
                            calc_dist_cats, num_classes=num_classes)
                    
    elif 'clinical' in model_name:
        if model_name == 'clinical_default':
            if options.dataset in ['four_classes_features_pathology']:
                breast_density_cats = 4
                mass_shape_cats= 8
                mass_margins_cats = 5
                calc_type_cats = 14
                calc_dist_cats = 5

                model_ft = Clinical_Model(model_name,
                                        input_vector_dim=\
                                        breast_density_cats+\
                                        mass_shape_cats+\
                                        mass_margins_cats+\
                                        calc_type_cats+\
                                        calc_dist_cats,
                                        num_classes=num_classes)
            
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
