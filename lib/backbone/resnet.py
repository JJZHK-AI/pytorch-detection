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


@BACKBONE_ZOO.register()
def yolov2_resnet50(cfg: DetectConfig):
    return ResNet(Bottleneck, [3, 4, 6, 3])


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


class ResNet(torch.nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             torch.nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride),
                                     padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * 4)
        self.relu = torch.nn.ReLU(inplace=True)
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