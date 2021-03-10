from optparse import OptionParser

parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=64, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=100, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vs', '--valid_size', dest='valid_size', type=float, default=0.1,
                  help='Proportion of training set used for validation')
parser.add_option('-j', '--workers', dest='workers', default=4, type='int',
                  help='number of subprocesses to use for data loading (default: 16)')

# For data
parser.add_option('--ddir', '--data_dir', dest='data_dir', default='./data',
                  help='Directory in which data is stored (default: ./data)')
parser.add_option('--dn', '--data_name', dest='data_name', default='cifar10',
                  help='mnist, cifar10, cifar100, fashion_mnist, svhn, smallnorb (default: mnist)')
parser.add_option('--rs', '--random_seed', dest='random_seed', default=2018, type='int',
                  help='Seed to ensure reproducibility (default: 2018)')

parser.add_option('--ih', '--img_h', dest='img_h', default=32, type='int',
                  help='input image height (default: 28)')
parser.add_option('--iw', '--img_w', dest='img_w', default=32, type='int',
                  help='input image width (default: 28)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 1)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=10, type='int',
                  help='number of classes (default: 10)')
parser.add_option('--exp', '--exp', type=str, default='full',
                  help="viewpoint exp name (NULL, azimuth, elevation, full)")
parser.add_option('--fam', '--familiar', dest='familiar', default=True,
                  help="viewpoint exp setting (novel, familiar)")

# Pick the loss type
parser.add_option('--lt', '--loss_type', dest='loss_type', default='cross-entropy',
                  help='margin, spread, cross-entropy, nll (default: margin)')

# For Margin loss
parser.add_option('--mp', '--m_plus', dest='m_plus', default=0.9, type='float',
                  help='m+ parameter (default: 0.9)')
parser.add_option('--mm', '--m_minus', dest='m_minus', default=0.1, type='float',
                  help='m- parameter (default: 0.1)')
parser.add_option('--la', '--lambda_val', dest='lambda_val', default=0.5, type='float',
                  help='Down-weighting parameter for the absent class (default: 0.5)')
parser.add_option('--al', '--alpha', dest='alpha', default=0.0005, type='float',
                  help='Regularization coefficient to scale down the reconstruction loss (default: 0.0005)')

# For Spread loss
parser.add_option('--mmin', '--m_min', dest='m_min', default=0.2, type='float',
                  help='m_min parameter (default: 0.2)')
parser.add_option('--mmax', '--m_max', dest='m_max', default=0.9, type='float',
                  help='m_min parameter (default: 0.9)')
parser.add_option('--nefm', '--n_eps_for_m', dest='n_eps_for_m', default=5, type='int',
                  help='number of epochs to increment the margin (default: 5)')
parser.add_option('--md', '--m_delta', dest='m_delta', default=0.1, type='float',
                  help='margin increment (default: 0.1)')

# For optimizer
parser.add_option('--lr', '--lr', dest='lr', default=0.1, type='float',
                  help='learning rate (default: 0.001)')
parser.add_option('--wd', '--weight_decay', dest='weight_decay', default=5e-4, type='float',
                  help='weight decay (default: 0.)')
parser.add_option('--beta1', '--beta1', dest='beta1', default=0.9, type='float',
                  help='beta 1 for Adam optimizer (default: 0.9)')

# For CapsNet
parser.add_option('--bb', '--backbone', dest='backbone', default='resnet',
                  help='simple, convnet, resnet (default: resnet)')
parser.add_option('--rv', '--resnet_version', dest='resnet_version', default='resnet20',
                  help='Version of resnet if used as backbone {resnet20, resnet32}(default: resnet20)')
parser.add_option('--rout', '--routing', dest='routing', default='TR',
                  help='Routing type: {DR, EM, SR, TR} (default: TR)')
parser.add_option('--nri', '--num_rout_iters', dest='num_rout_iters', default=3, type='int',
                  help='number of routing iterations for DR and EM (default: 3)')
parser.add_option('--nh', '--num_heads', dest='num_heads', default=1, type='int',
                  help='number of attention heads for TR (default: 1)')

parser.add_option('--A', '--A', dest='A', default=32, type='int',
                  help='dimension of each primary capsule (default: 32)')
parser.add_option('--B', '--B', dest='B', default=32, type='int',
                  help='dimension of each primary capsule (default: 32)')
parser.add_option('--C', '--C', dest='C', default=32, type='int',
                  help='dimension of each primary capsule (default: 32)')
parser.add_option('--D', '--D', dest='D', default=32, type='int',
                  help='dimension of each primary capsule (default: 32)')

# For save and loading
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/capsnet_transformer_routing/save/'
                          '20200817_100904_TR_resnet_smallnorb/models/2627.ckpt',
                  help='path to load a .ckpt model')

options, _ = parser.parse_args()
