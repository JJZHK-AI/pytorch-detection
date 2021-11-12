"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: layers.py
@time: 2021-11-12 16:05:57
@desc: 
"""
import torch
import torch.nn.functional
from lib.backbone.conv2d import Conv2d
from lib.backbone.layer_zoo import  LAYER_ZOO
from lib.backbone.util import layer_to_config
import math
from lib.backbone.util import get_layer


def yolov2_create_modules(mdef, cfg, **kwargs):
    mdef_net = cfg['net']
    inFilters = mdef["inFilters"]
    index = mdef["index"]
    mdefsummary = kwargs['mdefsummary']
    img_size = mdef_net["imagesize"]
    output_filters = mdef['filter_list']

    routs = kwargs["routs"]
    module_list = kwargs["mlist"]

    filters = output_filters[-1]
    l = None

    if mdef['type'] == 'seq':
        l = torch.nn.Sequential()
        my_seq_layers = mdef['layers']
        for index, layer in enumerate(my_seq_layers):
            if layer['type'] == 'conv_bn_leaky_re_lu':
                l.add_module("%d" % index, get_layer(layer, inFilters=inFilters))
                filters = layer['filters']
            elif layer['type'] == 'maxpool':
                l.add_module("%d" % index, get_layer(layer, inFilters=filters, seq=1))

    return layer_to_config(mdef['name'], l), filters


@LAYER_ZOO.register()
def conv_bn_leaky_re_lu(layer, **kwargs):
    inFilters = kwargs['inFilters']
    filters = layer['filters']
    ksize = layer['size']
    padding = layer['padding'] if 'padding' in layer else 0
    return Conv_BN_LeakyReLU(inFilters, filters, ksize, padding)


@LAYER_ZOO.register(version=1)
def maxpool(layer, **kwargs):
    k = layer['size']  # kernel size
    stride = layer['stride']
    maxpool = torch.nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

    if 'seq' in kwargs and kwargs['seq'] == 1:
        return maxpool
    else:
        return layer_to_config(layer['name'], maxpool)


class Conv_BN_LeakyReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)