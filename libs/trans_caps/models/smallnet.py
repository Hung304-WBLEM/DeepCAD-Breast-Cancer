import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SmallNet(nn.Module):
    def __init__(self, n_in, n_out):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)

        self.apply(weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


