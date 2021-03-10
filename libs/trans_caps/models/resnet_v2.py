import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_backbone(nn.Module):
    def __init__(self, cl_input_channels=3, cl_num_filters=128,
                 cl_stride=2):
        super(resnet_backbone, self).__init__()
        self.in_planes = 64

        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1),  # num_blocks=2 or 3
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=4, stride=cl_stride),  # num_blocks=2 or 4
        )

    def forward(self, x):
        out = self.pre_caps(x)      # x is an image
        return out
