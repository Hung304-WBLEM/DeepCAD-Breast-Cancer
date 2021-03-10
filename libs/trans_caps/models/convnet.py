import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ConvNet(nn.Module):
    def __init__(self, n_channels, planes):
        super(ConvNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*8),
            nn.ReLU(True),
        )

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)
        return out

