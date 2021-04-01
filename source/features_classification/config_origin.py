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
# parser.add_option("-f", "--freeze_type", dest="freeze_type",
#                     help="For Resnet50, freeze_type could be: 'none', 'all', 'last_fc', 'top1_conv_block', 'top2_conv_block', 'top3_conv_block'. For VGG16, freeze_type could be: 'none', 'all', 'last_fc', 'fc2', 'fc1', 'top1_conv_block', 'top2_conv_block'")

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



options, _ = parser.parse_args()
config['hyperparams'] = {}
for opt in vars(options):
    attr = getattr(options, opt)
    print(opt, attr, type(attr))

    config['hyperparams'][opt] = str(attr)

os.makedirs(options.save_path, exist_ok=True)
with open(os.path.join(options.save_path, 'config.ini'), 'w') as configfile:
    config.write(configfile)
