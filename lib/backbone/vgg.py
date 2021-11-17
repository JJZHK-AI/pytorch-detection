"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: vgg.py
@time: 2021-11-17 18:12:35
@desc: 
"""
import torch
from jjzhk.config import DetectConfig
from lib.backbone.backbone_layer import BACKBONE_ZOO

@BACKBONE_ZOO.register()
def vgg16(cfg: DetectConfig):
    return [
        torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
        torch.nn.LeakyReLU(negative_slope=0.1),
        torch.nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.1)
    ]


