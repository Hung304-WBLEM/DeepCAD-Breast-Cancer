import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import numpy as np


class CapsuleNet(nn.Module):
    def __init__(self, args, K=3, P=4):
        super(CapsuleNet, self).__init__()
        self.args = args
        A = args.A
        if args.backbone == 'simple':
            self.pre_caps = SmallNet(n_in=args.img_c, n_out=A)
        elif args.backbone == 'convnet':
            self.pre_caps = ConvNet(args.img_c, int(A/8))
        elif args.backbone == 'resnet':
            # A = 64 if args.data_name != 'cifar100' else 128
            A = 64
            self.pre_caps = resnet.__dict__[args.resnet_version](in_channels=args.img_c, planes=int(A/4))
        elif args.backbone == 'resnet_v2':
            self.pre_caps = resnet_backbone()
            A = 128

        kwargs = {}
        if args.routing == 'DR':
            self.mode = 'DR'
            from layers.dynamic_routing import PrimaryCaps, ConvCaps
            kwargs['iters'] = args.num_rout_iters

        elif args.routing == 'EM':
            self.mode = 'EM'
            from layers.em_routing import PrimaryCaps, ConvCaps
            kwargs['iters'] = args.num_rout_iters
            kwargs['final_lambda'] = 1e-2

        elif args.routing == 'SR':
            self.mode = 'SR'
            from layers.self_routing import PrimaryCaps, ConvCaps

        elif args.routing == 'TR':
            self.mode = 'TR'
            from layers.transformer_routing import PrimaryCaps, ConvCaps
            kwargs['num_heads'] = args.num_heads
            self.final_fc = nn.Linear(P * P, 1)
        else:
            raise Exception('Selected routing mechanism is not available!')

        self.primary_caps = PrimaryCaps(A, args.B, K=3, P=P, stride=1, pad=1)
        self.conv_caps1 = ConvCaps(args.B, args.C, K=3, P=P, stride=2, pad=1, **kwargs)
        # self.conv_caps2 = ConvCaps(args.C, args.D, K=3, P=P, stride=1, **kwargs)
        self.class_caps = ConvCaps(args.C, args.num_classes, K=4, P=P, pad=0, last_layer=True, **kwargs)

    def forward(self, imgs, y=None):
        x = self.pre_caps(imgs)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        # x = self.conv_caps2(x)

        if self.args.routing == 'DR':
            out = self.class_caps(x).squeeze()
            out = out.view(out.size(0), self.args.num_classes, -1)
            out = out.norm(dim=-1)
            out = out / out.sum(dim=1, keepdim=True)
            out = out.log()
        if self.args.routing == 'EM':
            a, _ = self.class_caps(x)
            out = a.view(a.size(0), -1)
            out = out / out.sum(dim=1, keepdim=True)
            out = out.log()
        elif self.args.routing == 'SR':
            a, _ = self.class_caps(x)
            out = a.view(a.size(0), -1)
            out = out.log()
        elif self.args.routing == 'TR':
            x = self.class_caps(x)
            x = x.view(imgs.shape[0], self.args.num_classes, -1)
            out = self.final_fc(x).squeeze(-1)

        _, y_pred = out.max(dim=1)
        y_pred_ohe = F.one_hot(y_pred, self.args.num_classes)

        return y_pred_ohe, out

