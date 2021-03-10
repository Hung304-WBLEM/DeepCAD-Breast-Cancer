import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, pad=1):
        super(PrimaryCaps, self).__init__()
        self.conv_pose = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                                   kernel_size=K, stride=stride, padding=pad, bias=False)
        self.conv_a = nn.Conv2d(in_channels=A, out_channels=B,
                                kernel_size=K, stride=stride, padding=pad, bias=False)
        self.psize = P * P
        self.B = B
        self.bn_a = nn.BatchNorm2d(B)
        self.bn_pose = nn.BatchNorm2d(B * P * P)

    def forward(self, x_in):
        a, pose = self.conv_a(x_in), self.conv_pose(x_in)
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)
        return a, pose


class ConvCaps(nn.Module):
    def __init__(self, A=32, B=32, K=3, P=4, stride=2, pad=0, last_layer=False):
        super(ConvCaps, self).__init__()
        self.A = A
        self.B = B
        self.C = P * P
        self.D = P * P if not last_layer else 1

        self.k = K
        self.kk = K * K
        self.kkA = K * K * A
        self.stride = stride
        self.pad = pad
        self.last_layer = last_layer

        if not self.last_layer:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B * self.D, self.C))
            nn.init.kaiming_uniform_(self.W1.data)
            self.pose_bn = nn.BatchNorm2d(B * self.D)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, self.C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

    def forward(self, x):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        a, pose = x
        b, _, h, w = a.shape

        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        if hasattr(self, 'W1'):
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)

        # [b, kkA, l]
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1, 2)

        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1, 2)

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        if hasattr(self, 'W1'):
            pose_out = self.pose_bn(pose_out.view(b, -1, oh, ow))
        else:
            pose_out = None

        return a_out, pose_out
