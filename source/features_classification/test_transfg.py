# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from features_classification.datasets import Mass_Shape_Dataset, Mass_Margins_Dataset
from features_classification.datasets import Calc_Type_Dataset, Calc_Dist_Dataset
from sklearn.preprocessing import label_binarize

from TransFG.models.modeling import VisionTransformer, CONFIGS
from TransFG.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from TransFG.utils.data_utils import get_loader
from TransFG.utils.dist_util import get_world_size
from TransFG.train import setup
from TransFG.train import set_seed
from TransFG.train import predict

from features_classification.train_utils import add_pr_curve_tensorboard
from features_classification.eval_utils import eval_all
from features_classification.eval_utils import eval_all, evalplot_precision_recall_curve, evalplot_roc_curve, evalplot_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["Mass_Shape", "Mass_Margins", "Calc_Type", "Calc_Dist", "CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--save_path", type=str,
                        help="The path to saved experiment.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    parser.add_argument("--crt", "--criterion", dest="criterion", type=str, default="ce",
                      help="Choose criterion: ce, bce")
    parser.add_argument("--wc", "--weighted_classes", dest="weighted_classes",
                    default=False, action='store_true',
                    help="enable if you want to train with classes weighting")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # save_path = '/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm/mass_shape_comb_feats_omit/transfg_b16_e100_224x224'
    save_path = args.save_path
    writer = SummaryWriter(log_dir=os.path.join(save_path, "tensorboard_logs"))

    input_size = args.img_size
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    root_dir = '/home/hqvo2/Projects/Breast_Cancer/data/processed_data2'

    if args.dataset == 'Mass_Shape':
        mass_shape_dir = 'mass/cls/mass_shape_comb_feats_omit'
        testset = Mass_Shape_Dataset(os.path.join(root_dir, mass_shape_dir, 'test'),
                                        data_transforms['test'])
        classes = Mass_Shape_Dataset.classes
    elif args.dataset == 'Mass_Margins':
        mass_margins_dir = 'mass/cls/mass_margins_comb_feats_omit'
        testset = Mass_Margins_Dataset(os.path.join(root_dir, mass_margins_dir, 'test'),
                                        data_transforms['test'])
        classes = Mass_Margins_Dataset.classes

    elif args.dataset == 'Calc_Type':
        calc_type_dir = 'calc/cls/calc_type_comb_feats_omit'
        testset = Calc_Type_Dataset(os.path.join(root_dir, calc_type_dir, 'test'),
                                        data_transforms['test'])
        classes = Calc_Type_Dataset.classes

    elif args.dataset == 'Calc_Dist':
        calc_dist_dir = 'calc/cls/calc_dist_comb_feats_omit'
        testset = Calc_Dist_Dataset(os.path.join(root_dir, calc_dist_dir, 'test'),
                                        data_transforms['test'])
        classes = Calc_Dist_Dataset.classes
        
        
        
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    preds, labels, _ = predict(args, model, writer, test_loader)

    num_classes = 8
    with torch.no_grad():
        softmaxs = torch.softmax(torch.tensor(preds), dim=-1)
        binarized_labels = label_binarize(
            labels, classes=[*range(num_classes)])

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(writer, classes, i, np.array(labels), softmaxs.cpu().detach().numpy())

    # my roc curve
    y_true = np.array(labels) 
    y_proba_pred = softmaxs.cpu().detach().numpy()
    binarized_y_true = label_binarize(y_true, classes=[*range(len(classes))])
    y_pred = y_proba_pred.argmax(axis=1)

    writer.add_figure('test confusion matrix',
                      evalplot_confusion_matrix(y_true,
                                                y_pred, classes, fig_only=True),
                        global_step=None)
    writer.add_figure('test roc curve',
                        evalplot_roc_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)
    writer.add_figure('test pr curve',
                        evalplot_precision_recall_curve(binarized_y_true,
                                           y_proba_pred, classes, fig_only=True),
                        global_step=None)
    
    # evaluation
    eval_all(np.array(labels),
             softmaxs.cpu().detach().numpy(), classes, save_path)




if __name__ == '__main__':
    main()
