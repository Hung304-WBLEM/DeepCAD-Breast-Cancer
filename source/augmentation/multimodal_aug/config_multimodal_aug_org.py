import os
import configparser

from configparser import ConfigParser
from optparse import OptionParser


config = ConfigParser()
parser = OptionParser()


parser.add_option("--ngpu", dest="ngpu",
                  type=int, default=2,
                  help='Numbers of used GPU')
parser.add_option("--njobs", dest="num_workers", type=int,
                  default=0)
parser.add_option("-b", "--batch_size", dest='batch_size', type=int, default=64,
                  help="Batch size for training")
parser.add_option("-e", "--epochs", type=int, default=20,
                  help="the number of epochs for training")


parser.add_option("--save", action='store_true')
parser.add_option("-s", "--save_path",
                    help="Path for logging and saving trained models")


parser.add_option("--ims", "--image_size", dest="image_size",
                  type=int, default=64,
                  help='Size of the input images')
parser.add_option("--nz", dest="latent_size",
                  type=int, default=100,
                  help='Size of the latent vectors')


parser.add_option("--lsmooth", dest="random_lbl_smooth",
                  action='store_true')

parser.add_option("--rl_min", "--real_label_min", dest="real_label_min",
                  type=float, default=1.0,
                  help='Set min value for real label')
parser.add_option("--rl_max", "--real_label_max", dest="real_label_max",
                  type=float, default=1.0,
                  help='Set max value for real label')

parser.add_option("--fl_min", "--fake_label_min", dest="fake_label_min",
                  type=float, default=0.0,
                  help='Set min value for fake label')
parser.add_option("--fl_max", "--fake_label_max", dest="fake_label_max",
                  type=float, default=0.0,
                  help='Set max value for fake label')

parser.add_option("--d_lr", "--discriminator_learning_rate", dest="disc_lr",
                  type=float, default=0.0004,
                  help="Learning rate of the Discriminator")
parser.add_option("--g_lr", "--generator_learning_rate", dest="gen_lr",
                  type=float, default=0.0002,
                  help="Learning rate of the Generator")
parser.add_option("--beta1", dest="beta1",
                  type=float, default=0.5,
                  help="Beta1 for the Adam optimizer")

parser.add_option("--loss_func", dest="loss_func",
                  type=str, default='minmax',
                  help="Select loss function: 1. minmax; 2. wasserstein")
parser.add_option("--d_num_iters", dest="d_num_iters",
                  type=int, default=1,
                  help="The number of forward-backward runs for discriminator in each iteration")
parser.add_option("--d_fuse_type", dest="disc_fuse_type",
                  type=str, default='concat',
                  help="Fusion type for Multi-modal discriminator")


options, _ = parser.parse_args()
config['hyperparams'] = {}
for opt in vars(options):
    attr = getattr(options, opt)
    print(opt, attr, type(attr))

    config['hyperparams'][opt] = str(attr)

if options.save is True:
    os.makedirs(options.save_path, exist_ok=True)
    with open(os.path.join(options.save_path, 'config.ini'), 'w') as configfile:
        config.write(configfile)
