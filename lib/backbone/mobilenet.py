"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: mobilenet.py
@time: 2021-11-17 20:32:14
@desc: 
"""
import torch
from lib.backbone.backbone_layer import BACKBONE_ZOO
from jjzhk.config import DetectConfig

@BACKBONE_ZOO.register()
def mobilenetv1(cfg: DetectConfig):
    layers = []

    layers += [Mobilenet_conv_bn(3, 32, 2)]
    layers += [Mobilenet_conv_dw(32, 64, 1)]
    layers += [Mobilenet_conv_dw(64, 128, 2)]
    layers += [Mobilenet_conv_dw(128, 128, 1)]
    layers += [Mobilenet_conv_dw(128, 256, 2)]

    layers += [Mobilenet_conv_dw(256, 256, 1)]
    layers += [Mobilenet_conv_dw(256, 512, 2)]

    layers += [Mobilenet_conv_dw(512, 512, 1)]
    layers += [Mobilenet_conv_dw(512, 512, 1)]
    layers += [Mobilenet_conv_dw(512, 512, 1)]
    layers += [Mobilenet_conv_dw(512, 512, 1)]
    layers += [Mobilenet_conv_dw(512, 512, 1)]

    layers += [Mobilenet_conv_dw(512, 1024, 2)]
    layers += [Mobilenet_conv_dw(1024, 1024, 1)]
    return layers

@BACKBONE_ZOO.register()
def mobilenetv2(cfg: DetectConfig):
    layers = []

    layers += [Mobilenet_conv_bn(3, 32, 2)]
    layers += [Mobilenet_inverted_residual_bottleneck(32, 16, 1, 1)]
    layers += [Mobilenet_inverted_residual_bottleneck(16, 24, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(24, 24, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(24, 32, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(32, 32, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(32, 32, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(32, 64, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(64, 64, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(64, 64, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(64, 64, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(64, 96, 1, 6)]

    layers += [Mobilenet_inverted_residual_bottleneck(96, 96, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(96, 96, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(96, 160, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(160, 160, 1, 6)]

    layers += [Mobilenet_inverted_residual_bottleneck(160, 160, 1, 6)]
    layers += [Mobilenet_inverted_residual_bottleneck(160, 320, 1, 6)]
    return layers


class Mobilenet_conv_bn(torch.nn.Module):
    def __init__(self, inp, oup, stride):
        super(Mobilenet_conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(oup),
            torch.nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class Mobilenet_conv_dw(torch.nn.Module):
    def __init__(self, inp, oup, stride):
        super(Mobilenet_conv_dw, self).__init__()
        self.conv = torch.nn.Sequential(
            # dw
            torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            torch.nn.BatchNorm2d(inp),
            torch.nn.ReLU(inplace=True),
            # pw
            torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
            torch.nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class Mobilenet_inverted_residual_bottleneck(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(Mobilenet_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = torch.nn.Sequential(
            # pw
            torch.nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(inp * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # dw
            torch.nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            torch.nn.BatchNorm2d(inp * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # pw-linear
            torch.nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        )
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
