import os

from configparser import ConfigParser
from optparse import OptionParser

config = ConfigParser()
parser = OptionParser()

parser.add_option("-d", "--dataset",
                    help="Name of the available datasets")
parser.add_option("-s", "--save_path",
                    help="Path to save the trained model")
parser.add_option("-m", "--model_name",
                    help="Select the backbone for training. Available backbones include: 'resnet', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'")
parser.add_option("-b", "--batch_size", type=int,
                    help="Batch size for training")
parser.add_option("-e", "--epochs", type=int,
                    help="the number of epochs for training")
parser.add_option("-i", "--input_size", type=int,
                  help="resize the input image")
parser.add_option("--wc", "--weighted_classes", dest="weighted_classes",
                    default=False, action='store_true',
                    help="enable if you want to train with classes weighting")
parser.add_option("--ws", "--weighted_samples", dest="weighted_samples",
                    default=False, action='store_true',
                    help="enable if you want to weight samples within batch")
parser.add_option("--lr", "--learning_rate", dest="learning_rate", type=float,
                    help="Learning rate")
parser.add_option("--wd", "--weights_decay", dest="weights_decay", type=float, default=0,
                    help="Weights decay")
parser.add_option("--opt", "--optimizer", dest="optimizer", type=str,
                    help="Choose optimizer: sgd, adam")
parser.add_option("--crt", "--criterion", dest="criterion", type=str, default="ce",
                  help="Choose criterion: ce, bce")
parser.add_option("--aug_type", dest="augmentation_type", type=str,
                  default="torch", help="Choose augmentation type. Available augmentation types include: torch, albumentations")
parser.add_option("--train_uct", "--train_uncertainty", dest="train_uncertainty", type=float,
                  default=0.0, help="Choose uncertainty thresholds to train without input features")
parser.add_option("--test_uct", "--test_uncertainty", dest="test_uncertainty", type=float,
                  default=0.0, help="Choose uncertainty thresholds to test without input features")
parser.add_option("--njobs", dest="num_workers", type=int,
                  default=0)

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

options, _ = parser.parse_args()
config['hyperparams'] = {}
for opt in vars(options):
    attr = getattr(options, opt)
    print(opt, attr, type(attr))

    config['hyperparams'][opt] = str(attr)

os.makedirs(options.save_path, exist_ok=True)
with open(os.path.join(options.save_path, 'config.ini'), 'w') as configfile:
    config.write(configfile)
