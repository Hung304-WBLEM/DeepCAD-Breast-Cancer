import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, pad=1):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                              kernel_size=K, stride=stride, padding=pad, bias=False)
        self.psize = P * P
        self.B = B
        self.nonlinear_act = nn.BatchNorm2d(self.psize*B)

    def forward(self, x_in):
        pose = self.nonlinear_act(self.conv(x_in))
        return pose


class ConvCaps(nn.Module):
    def __init__(self, A=32, B=32, K=3, P=4, stride=2, pad=0, **kwargs):
        super(ConvCaps, self).__init__()
        self.A = A
        self.B = B
        self.k = K
        self.kk = K * K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.kkA = K * K * A
        self.pad = pad
        # self.W = nn.Parameter(torch.randn(self.kkA, B, self.P, self.P))

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.P, self.P))
        nn.init.kaiming_uniform_(self.W.data)

        num_heads = kwargs['num_heads']
        self.router = TransformerRouter(num_ind=B, num_heads=num_heads, dim=self.psize)

    def forward(self, x):
        b, c, h, w = x.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(x, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(b, l, self.kkA, self.P, self.P).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b*l, kkA, B, psize]
        v = pose_out.view(b * l, self.kkA, self.B, self.psize)

        # [b*l, B, psize]
        pose_out = self.router(v)

        # [b, l, B*psize]
        pose_out = pose_out.view(b, l, self.B*self.psize)

        oh = ow = math.floor(l**(1/2))

        pose_out = pose_out.view(b, oh, ow, self.B * self.psize)
        return pose_out.permute(0, 3, 1, 2)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.M = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.kaiming_uniform_(self.M.data)

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.kaiming_uniform_(self.S.data)

        self.p = nn.Parameter(torch.Tensor(1, num_seeds))
        nn.init.xavier_uniform_(self.p)

        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        if ln:
            self.ln0 = nn.BatchNorm1d(num_seeds*dim)
            self.ln1 = nn.BatchNorm1d(num_seeds*dim)
        self.fc_o = nn.Linear(dim, dim)

    def log_prob(self, x, mu, sigma, p):
        diff = x - mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        ll = -0.5 * math.log(2 * math.pi) - sigma.log() - 0.5 * (diff ** 2 / sigma ** 2)
        ll = ll.sum(-1)
        # ll = ll + (p + 1e-10).log().unsqueeze(-2)
        ll = ll + p.log().unsqueeze(-2)
        return ll

    def forward(self, X):
        b, _, num_caps, dim = X.shape
        dim_split = self.dim // self.num_heads

        M = self.M.repeat(X.size(0), 1, 1)
        sigma = F.softplus(self.S.repeat(b, 1, 1))
        p = torch.softmax(self.p.repeat(b, 1), -1)
        K, V = self.fc_k(X), self.fc_v(X)

        # divide for multi-head purpose
        mu_ = torch.cat(M.split(dim_split, 2), 0)
        sigma_ = torch.cat(sigma.split(dim_split, 2), 0)
        p_ = p.repeat(self.num_heads, 1)
        K_ = torch.cat(K.split(dim_split, 3), 0)
        V_ = torch.cat(V.split(dim_split, 3), 0)

        ll = self.log_prob(K_, mu_, sigma_, p_)
        A = torch.softmax(ll / math.sqrt(self.dim), 1).unsqueeze(-1)
        O = torch.cat((mu_ + torch.sum(A * V_, dim=1)).split(b, 0), 2)

        O = self.ln0(O.view(b, num_caps*dim)).view(b, num_caps, dim)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O.view(b, num_caps*dim)).view(b, num_caps, dim)

        return O


class TransformerRouter(nn.Module):
    def __init__(self, num_ind, num_heads, dim=16, ln=True):
        super(TransformerRouter, self).__init__()
        self.pma = PMA(dim, num_heads, num_ind, ln=ln)

    def forward(self, pose):
        pose = self.pma(pose)
        return pose
