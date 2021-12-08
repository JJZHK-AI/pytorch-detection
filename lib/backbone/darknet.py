"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: darknet.py
@time: 2021-11-17 20:29:14
@desc: 
"""
import torch
from lib.backbone.backbone_layer import BACKBONE_ZOO
from jjzhk.config import DetectConfig

@BACKBONE_ZOO.register()
def darknet19(cfg: DetectConfig):
    layer = []
    layer += [Darknet_conv_bn(3, 32, 1)]
    layer += [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
    layer += [Darknet_conv_bn(32, 64, 1)]
    layer += [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]

    layer += [Darknet_conv_block(64, 128, 1)]
    layer += [Darknet_conv_block(128, 128, 1)]

    layer += [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]

    layer += [Darknet_conv_block(128, 256, 1)]
    layer += [Darknet_conv_block(256, 256, 1)]
    layer += [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]

    layer += [Darknet_conv_block(256, 512, 1)]
    layer += [Darknet_conv_block(512, 512, 1)]
    layer += [Darknet_conv_block(512, 512, 1)]
    layer += [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]

    layer += [Darknet_conv_block(512, 1024, 1)]
    layer += [Darknet_conv_block(1024, 1024, 1)]
    layer += [Darknet_conv_block(1024, 1024, 1)]

    return layer


@BACKBONE_ZOO.register()
def darknet53(cfg: DetectConfig):
    layer = []
    layer += [Darknet_conv_bn(3, 32, 1)]
    layer += [Darknet_conv_block(32, 64, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(64, 64, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(64, 128, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(128, 128, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(128, 128, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(128, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 256, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(256, 512, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(512, 512, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(512, 1024, stride=1, darknettype=53)]

    layer += [Darknet_conv_block(1024, 1024, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(1024, 1024, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(1024, 1024, stride=1, darknettype=53)]
    layer += [Darknet_conv_block(1024, 1024, stride=1, darknettype=53)]
    return layer

@BACKBONE_ZOO.register()
def yolov2_darknet19(cfg: DetectConfig):
    return DarkNet_19(cfg)


@BACKBONE_ZOO.register()
def yolov2_darknet19_tiny(cfg: DetectConfig):
    return DarkNet_Tiny(cfg)

@BACKBONE_ZOO.register()
def yolov3_darknet53(cfg: DetectConfig):
    return DarkNet_53(cfg)

@BACKBONE_ZOO.register()
def yolov3_tiny(cfg: DetectConfig):
    return DarkNet_Light(cfg)

@BACKBONE_ZOO.register()
def yolov3_spp(cfg: DetectConfig):
    return DarkNet_53(cfg)


class Darknet_conv_bn(torch.nn.Module):
    def __init__(self, inp, oup, stride):
        super(Darknet_conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inp, oup, kernel_size=(3, 3), stride=(stride, stride),
                            padding=(1, 1), bias=False),
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
                torch.nn.Conv2d(inp, depth, (1, 1), (1, 1), bias=False),
                torch.nn.BatchNorm2d(depth),
                torch.nn.LeakyReLU(0.1, inplace=True),
                torch.nn.Conv2d(depth, oup, (3, 3), (stride, stride), (1, 1), bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inp, oup, (3, 3), (stride, stride), (1, 1), bias=False),
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


class DarkNet_19(torch.nn.Module):
    def __init__(self, cfg: DetectConfig):
        super(DarkNet_19, self).__init__()
        self.cfg = cfg
        self.conv_1 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            torch.nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            torch.nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            torch.nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = torch.nn.MaxPool2d((2, 2), 2)
        self.conv_5 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = torch.nn.MaxPool2d((2, 2), 2)
        self.conv_6 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        C_4 = self.conv_4(x)
        C_5 = self.conv_5(self.maxpool_4(C_4))
        C_6 = self.conv_6(self.maxpool_5(C_5))

        # x = self.conv_7(C_6)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # return x
        return C_4, C_5, C_6


class Conv_BN_LeakyReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, ksize, padding=(padding, padding), stride=(stride, stride),
                            dilation=(dilation, dilation)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_Tiny(torch.nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super(DarkNet_Tiny, self).__init__()
        self.cfg = cfg
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.conv_1 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            Conv_BN_LeakyReLU(32, 32, 3, padding=1, stride=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            Conv_BN_LeakyReLU(64, 64, 3, padding=1, stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 128, 3, padding=1, stride=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 256, 3, padding=1, stride=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 512, 3, padding=1, stride=2),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        C_3 = self.conv_3(x)
        C_4 = self.conv_4(C_3)
        C_5 = self.conv_5(C_4)
        # x = self.conv_6(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return C_3, C_4, C_5


class DarkNet_53(torch.nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self, cfg, num_classes=1000):
        super(DarkNet_53, self).__init__()
        self.cfg = cfg
        # stride = 2
        self.layer_1 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = torch.nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        C_3 = self.layer_3(x)
        C_4 = self.layer_4(C_3)
        C_5 = self.layer_5(C_4)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return C_3, C_4, C_5


class resblock(torch.nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = torch.nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_Light(torch.nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super(DarkNet_Light, self).__init__()
        self.cfg = cfg
        # backbone network : DarkNet_Light
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = torch.nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = torch.nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = torch.nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = torch.nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = torch.nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = torch.nn.Sequential(
            torch.nn.ZeroPad2d((0, 1, 0, 1)),
            torch.nn.MaxPool2d((2, 2), 1)  # stride = 32
        )

        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)  # stride = 16
        x = self.maxpool_5(C_4)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)  # stride = 32

        return C_4, C_5

