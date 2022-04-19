import os
import configparser

from configparser import ConfigParser
from optparse import OptionParser

config = ConfigParser()
parser = OptionParser()

parser.add_option("--exp_name", dest='experiment_name',
                    help="Name of the experiment for mlflow")
parser.add_option("-d", "--dataset",
                    help="Name of the available datasets")
parser.add_option("--tr", "--train_rate", dest="train_rate", type=float, default=1,
                  help="Part of the dataset that you want to train")
parser.add_option("-m", "--model_name",
                    help="Select the backbone for training. Available backbones include: 'resnet', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'")
parser.add_option("--use_pretrained", dest="use_pretrained",
                  default=False, action='store_true',
                  help="Select if you want to use pretrained weights (random weights, otherwise)")
parser.add_option("-b", "--batch_size", type=int, default=32,
                    help="Batch size for training")
parser.add_option("-e", "--epochs", type=int, default=100,
                    help="the number of epochs for training")
parser.add_option("-i", "--input_size", type=int, default=224,
                  help="resize the input image")
parser.add_option("--wc", "--weighted_classes", dest="weighted_classes",
                    default=False, action='store_true',
                    help="enable if you want to train with classes weighting")
parser.add_option("--ws", "--weighted_samples", dest="weighted_samples",
                    default=False, action='store_true',
                    help="enable if you want to weight samples within batch")
parser.add_option("--lr", "--learning_rate", dest="learning_rate", type=float, default=0.01,
                    help="Learning rate") # deprecated
parser.add_option("--wd", "--weights_decay", dest="weights_decay", type=float, default=0,
                    help="Weights decay")
parser.add_option("--opt", "--optimizer", dest="optimizer", type=str,
                    help="Choose optimizer: sgd, adam")
parser.add_option("--use_lr_scheduler", dest="use_lr_scheduler",
                    default=False, action='store_true',
                    help="activate if you want to use learning rate scheduler")
parser.add_option("--use_clinical_feats", dest="use_clinical_feats",
                  default=False, action='store_true')
parser.add_option("--use_clinical_feats_only", dest="use_clinical_feats_only",
                  default=False, action='store_true')
parser.add_option("--fuse", default='concat',
                  help="Choose fusion methods. Available options include: 'concat', 'coatt', 'crossatt'", dest="fusion_type")

parser.add_option("--crt", "--criterion", dest="criterion", type=str, default="ce",
                  help="Choose criterion: ce, bce")
parser.add_option("--aug_type", dest="augmentation_type", type=str,
                  default="torch", help="Choose augmentation type. Available augmentation types include: torch, albumentations, augmix")
parser.add_option("--best_ckpt_metric", dest="best_ckpt_metric", 
                  default="acc", choices=['acc', 'macro_ap', 'micro_ap', 'macro_auc', 'micro_auc', 'macro_auc_only'],
                  help="Choose metric to select best model for ckpt")
parser.add_option("--ckpt", dest="ckpt_path",
                  default=None, help="Path to load the model checkpoint")
parser.add_option("--njobs", dest="num_workers", type=int,
                  default=0)

# Train one stage
parser.add_option("--one_stage_training", dest="one_stage_training",
                    default=False, action='store_true',
                    help="enable if you want to use conventional one-stage training")

# Three-stages training hyperparams
## 1st stage
parser.add_option("--first_stage_lr",
                  dest="first_stage_learning_rate", type=float, default=0.001)
parser.add_option("--first_stage_wd",
                  dest="first_stage_weight_decay", type=float, default=0.01)
parser.add_option("--first_stage_freeze",
                  dest="first_stage_last_frozen_layer", type=int, default=158)
## 2nd stage
parser.add_option("--second_stage_lr",
                  dest="second_stage_learning_rate", type=float, default=0.0001)
parser.add_option("--second_stage_wd",
                  dest="second_stage_weight_decay", type=float, default=0.01)
parser.add_option("--second_stage_freeze",
                  dest="second_stage_last_frozen_layer", type=int, default=87)
## 3rd stage
parser.add_option("--third_stage_lr",
                  dest="third_stage_learning_rate", type=float, default=0.00001)
parser.add_option("--third_stage_wd",
                  dest="third_stage_weight_decay", type=float, default=0.01)
parser.add_option("--third_stage_freeze",
                  dest="third_stage_last_frozen_layer", type=int, default=-1)
## 4th stage
parser.add_option("--is_fourth_stage", "--is_fourth_stage", dest="train_with_fourth_stage",
                    default=False, action='store_true',
                    help="enable if you want to train with 4th stage")
parser.add_option("--fourth_stage_lr",
                  dest="fourth_stage_learning_rate", type=float, default=0.000001)
parser.add_option("--fourth_stage_wd",
                  dest="fourth_stage_weight_decay", type=float, default=0.01)
parser.add_option("--fourth_stage_freeze",
                  dest="fourth_stage_last_frozen_layer", type=int, default=-1)

# Dilated Convolution
parser.add_option("--rnet_dil_2nd",
                  dest="resnet_dilated_layer2",
                  default=False, action='store_true',
                  help="enable if you want to dilate layer2 in the torchvision resnet model")
parser.add_option("--rnet_dil_3rd",
                  dest="resnet_dilated_layer3",
                  default=False, action='store_true',
                  help="enable if you want to dilate layer3 in the torchvision resnet model")
parser.add_option("--rnet_dil_4th",
                  dest="resnet_dilated_layer4",
                  default=False, action='store_true',
                  help="enable if you want to dilate layer4 in the torchvision resnet model")

# Ranking Hinge Loss hyperparams
parser.add_option("--sim_func",
                  dest="sim_func", choices=['dot', 'cosine', 'l2'],
                  default='dot', help="Choose Similarity Function")
parser.add_option("--margin",
                  dest="margin", type=float, default=3,
                  help="Hinge margin")

options, _ = parser.parse_args()
# config['hyperparams'] = {}
# for opt in vars(options):
#     attr = getattr(options, opt)
#     print(opt, attr, type(attr))
# 
#     config['hyperparams'][opt] = str(attr)

# if options.save is True:
#     os.makedirs(options.save_path, exist_ok=True)
#     with open(os.path.join(options.save_path, 'config.ini'), 'w') as configfile:
#         config.write(configfile)


def read_config(config_path):
    print(f'{config_path} exists?', os.path.exists(config_path))
    saved_config = ConfigParser(allow_no_value=True)

    saved_config.read(config_path)

    
    for opt in vars(options):
        attr = getattr(options, opt)
        keep_default = False

        try:
            if type(attr) is int:
                saved_attr = saved_config.getint('hyperparams', opt)
            elif type(attr) is float:
                saved_attr = saved_config.getfloat('hyperparams', opt)
            elif type(attr) is bool:
                saved_attr = saved_config.getboolean('hyperparams', opt)
            else:
                saved_attr = saved_config.get('hyperparams', opt)
        except ValueError: # For in the case when of the hyperparam 'learning_rate'
            saved_attr = None
        except configparser.NoOptionError:
            # For handling when saved options file
            # do not have some fields
            keep_default = True
            
        # print(type(attr), type(saved_attr))

        if keep_default is False:
            if type(saved_attr) is type(None):
                setattr(options, opt, None)
            else:
                setattr(options, opt, saved_attr)


    return options


if __name__ == '__main__':
    print('MAIN')
    saved_options = read_config(config_path='/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/four_classes_mass_calc_pathology/r50_b32_e100_224x224_adam_wc_ws_Thu Apr  1 15:52:21 CDT 2021/config.ini')
    for opt in vars(saved_options):
        attr = getattr(saved_options, opt)
        print(opt, attr, type(attr))

