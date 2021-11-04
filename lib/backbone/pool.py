"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: pool.py
@time: 2021-09-28 13:49:43
@desc: 
"""
import torch
from lib.backbone.layer_zoo import LAYER_ZOO


@LAYER_ZOO.register()
def maxpool(layer, **kwargs):
    list = []
    k = layer['size']  # kernel size
    stride = layer['stride']
    maxpool = torch.nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
    if k == 2 and stride == 1:  # yolov3-tiny
        list.append(torch.nn.ZeroPad2d((0, 1, 0, 1)))
        list.append(maxpool)
    else:
        list.append(maxpool)
    return list
