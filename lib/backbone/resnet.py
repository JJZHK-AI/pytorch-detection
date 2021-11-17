"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: resnet.py
@time: 2021-11-17 20:21:58
@desc: 
"""
import torch
from lib.backbone.backbone_layer import BACKBONE_ZOO
from jjzhk.config import DetectConfig


@BACKBONE_ZOO.register()
def resnet50(cfg: DetectConfig):
    layers = []
    layers += [torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, padding_mode='zeros')]
    layers += [torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [torch.nn.ReLU(inplace=True)]
    layers += [torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)]
    layers += [Resnet_Bottleneck(64, 64, stride=1,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]
    layers += [Resnet_Bottleneck(256, 64, stride=1)]
    layers += [Resnet_Bottleneck(256, 64, stride=1)]

    layers += [Resnet_Bottleneck(256, 128, stride=2,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]

    layers += [Resnet_Bottleneck(512, 256, stride=2,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]

    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]

    return layers


@BACKBONE_ZOO.register()
def resnet152(cfg: DetectConfig):
    layers = []

    layers += [
        torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, padding_mode='zeros')]
    layers += [torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [torch.nn.LeakyReLU(negative_slope=0.1)]
    layers += [torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)]

    layers += [Resnet_Bottleneck(64, 64, stride=1,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]
    layers += [Resnet_Bottleneck(256, 64, stride=1)]
    layers += [Resnet_Bottleneck(256, 64, stride=1)]
    layers += [Resnet_Bottleneck(256, 128, stride=2,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]
    layers += [Resnet_Bottleneck(512, 128, stride=1)]

    layers += [Resnet_Bottleneck(512, 256, stride=2,
                                 downsample=torch.nn.Sequential(*[
                                     torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False,
                                                     padding_mode='zeros'),
                                     torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)
                                 ]))]

    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]
    layers += [Resnet_Bottleneck(1024, 256, stride=1)]

    return layers


class Resnet_Bottleneck(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None):
        super(Resnet_Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3   = torch.nn.BatchNorm2d(planes * expansion)
        self.relu  = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out