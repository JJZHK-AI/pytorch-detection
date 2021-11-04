"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: conv2d.py
@time: 2021-09-27 14:21:09
@desc: 
"""
import torch
from .util import layer_to_config
from .layer_zoo import LAYER_ZOO
from lib.backbone.layers import ModelLayer


@LAYER_ZOO.register()
def convolutional(layer, **kwargs):
    return Conv2d(layer, **kwargs).layers()


class Conv2d(ModelLayer):
    def __init__(self, layer_config, **kwargs):
        super(Conv2d, self).__init__(layer_config, **kwargs)

    def transfer_config(self, **kwargs):
        list = []
        bn = self._layer_config_['batch_normalize']
        filters = self._layer_config_['filters']
        k = self._layer_config_['size']  # kernel size
        stride = self._layer_config_['stride'] if 'stride' in self._layer_config_ else (
        self._layer_config_['stride_y'], self._layer_config_['stride_x'])
        in_filter = self._layer_config_["inFilters"]
        routs = kwargs['routs']

        if isinstance(k, int):  # single-size conv
            list.append(layer_to_config("Conv2d", torch.nn.Conv2d(in_channels=in_filter, out_channels=filters,
                                        kernel_size=(k, k), stride=stride,
                                        padding=self._layer_config_['pad'] if 'pad' in self._layer_config_ else 0,
                                        groups=self._layer_config_['groups'] if 'groups' in self._layer_config_ else 1,
                                        dilation=self._layer_config_[
                                            'dilation'] if 'dilation' in self._layer_config_ else 1,
                                        bias=not bn)))

        if bn:
            list.append(layer_to_config("BatchNorm2d", torch.nn.BatchNorm2d(filters))) # detection output (goes into yolo layer)
        else:
            routs.append(self._layer_config_['index'])

        if self._layer_config_['activation'] == 'leaky':
            list.append(layer_to_config("activation", torch.nn.LeakyReLU(0.1, inplace=False)))
        elif self._layer_config_['activation'] == 'relu':
            list.append(layer_to_config("activation", torch.nn.ReLU(inplace=True)))

        return list


