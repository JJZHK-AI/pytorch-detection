"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: layers.py
@time: 2021-09-28 13:51:50
@desc: 
"""
import torch
from lib.backbone.util import get_layer
from jjzhk.config import DetectConfig
from lib.backbone.layer_zoo import LAYER_ZOO


def ssd_create_modules(mdef, cfg: DetectConfig, **kwargs):
    type = mdef['type']
    infilters = mdef['inFilters']
    routs = kwargs['routs']
    l = None
    filters = 0
    if type in ['resblock', 'darknetconv', 'mobilenetconv', 'mobilenetdw', 'mobilenetblock']:
        filters = mdef['filters']
        l = get_layer(mdef, in_filter=infilters, routs=routs)
    elif type == 'darknetblock':
        darknettype = cfg['net']['darknettype']
        filters = mdef['filters']
        l = get_layer(mdef, in_filter=infilters, darknettype=darknettype, routs=routs)
    else:
        print('Warning: Unrecognized Layer Type: ' + mdef['type'])
    return l, filters


@LAYER_ZOO.register()
def resblock(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    planes = layer['planes']
    down = layer['down'] if 'down' in layer else 0
    routs = kwargs['routs']
    layer_index = layer['index']
    routs.append(layer_index)
    return Resnet_Bottleneck(in_filter, planes, stride, False if down == 0 else 1)


@LAYER_ZOO.register()
def darknetconv(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    filters = layer['filters']
    routs = kwargs['routs']
    layer_index = layer['index']
    routs.append(layer_index)

    return Darknet_conv_bn(in_filter, filters, stride)


@LAYER_ZOO.register()
def darknetblock(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    filters = layer['filters']
    routs = kwargs['routs']
    layer_index = layer['index']
    darknettype = kwargs['darknettype']
    routs.append(layer_index)
    return Darknet_conv_block(in_filter, filters, stride, darknettype=darknettype)


@LAYER_ZOO.register()
def mobilenetconv(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    filters = layer['filters']
    routs = kwargs['routs']
    layer_index = layer['index']
    routs.append(layer_index)

    return Mobilenet_conv_bn(in_filter, filters, stride)


@LAYER_ZOO.register()
def mobilenetdw(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    filters = layer['filters']
    routs = kwargs['routs']
    layer_index = layer['index']
    routs.append(layer_index)

    return Mobilenet_conv_dw(in_filter, filters, stride)


@LAYER_ZOO.register()
def mobilenetblock(layer, **kwargs):
    in_filter = kwargs["in_filter"]
    stride = layer['stride']
    filters = layer['filters']
    routs = kwargs['routs']
    layer_index = layer['index']
    ratio = layer['ratio']
    routs.append(layer_index)

    return Mobilenet_inverted_residual_bottleneck(in_filter, filters, stride, ratio)


class Resnet_Bottleneck(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(Resnet_Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = torch.nn.BatchNorm2d(planes * 4)
        self.relu  = torch.nn.ReLU(inplace=True)
        if downsample:
            self.downsample = torch.nn.Sequential()
            self.downsample.add_module("0", torch.nn.Conv2d(inplanes, planes * 4,
                                                            kernel_size=1, stride=stride, bias=False))
            self.downsample.add_module("1", torch.nn.BatchNorm2d(planes * 4))
        else:
            self.downsample = None
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


class Darknet_conv_bn(torch.nn.Module):
    def __init__(self, inp, oup, stride):
        super(Darknet_conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(oup),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class Darknet_conv_block(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=0.5, darknettype=19):
        super(Darknet_conv_block, self).__init__()
        self.darknet_type = darknettype
        self.use_res_connect = stride == 1 and inp == oup
        if self.use_res_connect:
            depth = int(oup*expand_ratio)
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inp, depth, kernel_size=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(depth),
                torch.nn.LeakyReLU(0.1, inplace=True),
                torch.nn.Conv2d(depth, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.LeakyReLU(0.1, inplace=True),
            )
        self.depth = oup

    def forward(self, x):
        if self.darknet_type == 19:
            return self.conv(x)
        else:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


class Mobilenet_conv_bn(torch.nn.Module):
    def __init__(self, inp, oup, stride):
        super(Mobilenet_conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
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
            torch.nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1,
                            groups=inp, bias=False),
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
