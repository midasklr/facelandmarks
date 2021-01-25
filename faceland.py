# -*- coding: utf-8 -*-
# @Time : 2021/1/12 上午10:10 
# @Author : midaskong 
# @File : faceland.py 
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def group_conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class FaceLanndInference(nn.Module):
    def __init__(self):
        super(FaceLanndInference, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()  # 16*56*56
        self.relu = nn.ReLU(inplace=True)

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 2),
            Block(3, 16, 32, 16, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck2 = nn.Sequential(
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 64, 24, nn.ReLU(inplace=True), None, 1),
            Block(3, 24, 64, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(3, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(3, 40, 96, 40, hswish(), SeModule(40), 1),
            Block(3, 40, 96, 40, hswish(), SeModule(40), 1),
            Block(3, 40, 128, 48, hswish(), SeModule(48), 1),
            Block(3, 48, 128, 48, hswish(), SeModule(48), 1),
        )

        self.conv8 = nn.Conv2d(48, 48, 7, 1, 0, groups=48)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(48)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(120, 196)


    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))  # [16, 56, 56]
        out1 = self.bneck1(x)  # 16*28*28

        x = self.bneck2(out1)  # 14*14*24
        x1 = self.avg_pool1(x)  # [24, 1, 1]
        x1 = x1.view(x1.size(0), -1)  # 24

        x = self.bneck3(x)  # [48, 7, 7]
        x2 = self.avg_pool2(x)  # [48, 1, 1]
        x2 = x2.view(x2.size(0), -1)  # 48

        x3 = self.relu(self.conv8(x))  # [48, 1, 1]
        x3 = x3.view(x1.size(0), -1)  # 128

        multi_scale = torch.cat([x1, x2, x3], 1)  # 200
        landmarks = self.fc(multi_scale)  # (200, 196)

        return landmarks


