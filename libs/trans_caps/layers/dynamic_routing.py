import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.other_utils import squash


class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, pad=1):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                              kernel_size=K, stride=stride, padding=pad, bias=False)
        self.psize = P * P
        self.B = B
        self.bn_pose = nn.BatchNorm2d(B * P * P)

    def forward(self, x_in):
        pose = self.bn_pose(self.conv(x_in))
        b, c, h, w = pose.shape
        pose = pose.permute(0, 2, 3, 1).contiguous()
        pose = squash(pose.view(b, h, w, self.B, self.psize))
        pose = pose.view(b, h, w, -1)
        pose = pose.permute(0, 3, 1, 2)
        return pose


class ConvCaps(nn.Module):
    def __init__(self, A=32, B=32, K=3, P=4, stride=2, n_heads=1, pad=0, **kwargs):
        super(ConvCaps, self).__init__()
        self.A = A
        self.B = B
        self.k = K
        self.kk = K * K
        self.kkA = K * K * A
        self.P = P
        self.C = P * P
        self.D = P * P
        self.stride = stride
        self.pad = pad
        self.iters = kwargs['iters']
        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B*self.D, self.C))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, pose):
        # x: [b, AC, h, w]
        b, _, h, w = pose.shape
        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        # [b, l, kkA, BD]
        pose_out = torch.matmul(self.W, pose).squeeze(-1)
        # [b, l, kkA, B, D]
        pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B, 1]
        b = pose.new_zeros(b, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            c = torch.softmax(b, dim=3)

            # [b, l, 1, B, D]
            s = (c * pose_out).sum(dim=2, keepdim=True)
            # [b, l, 1, B, D]
            v = squash(s)

            b = b + (v * pose_out).sum(dim=-1, keepdim=True)

        # [b, l, B, D]
        v = v.squeeze(2)
        # [b, l, BD]
        v = v.view(v.shape[0], l, -1)
        # [b, BD, l]
        v = v.transpose(1, 2).contiguous()

        oh = ow = math.floor(l**(1/2))

        # [b, BD, oh, ow]
        return v.view(v.shape[0], -1, oh, ow)
