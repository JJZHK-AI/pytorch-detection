'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: fssd.py
@time: 2021-09-25 16:47:48
@desc: 
'''
import torch
from lib.model.base import DetectionModel
from lib.model.model_zoo import MODEL_ZOO


@MODEL_ZOO.register()
def fssd(cfg):
    return FSSD(cfg)


class FSSD(DetectionModel):
    def __init__(self, cfg):
        super(FSSD, self).__init__(cfg)
        extras, features, head = self._add_extras_(self.feature_layer,
                                                   self.number_box,
                                                   self.num_classes,
                                                   self.cfg['net']['lite'])

        self.base = torch.nn.ModuleList(self.backbone)

        self.extras = torch.nn.ModuleList(extras)
        self.feature_layer = self.feature_layer[0][0]
        self.transforms = torch.nn.ModuleList(features[0])
        self.pyramids = torch.nn.ModuleList(features[1])
        # Layer learns to scale the l2 normalized features from conv4_3
        self.norm = torch.nn.BatchNorm2d(int(self.cfg['net']['features'][0][1][-1] / 2) * len(self.transforms), affine=True)
        # print(self.extras)

        self.loc = torch.nn.ModuleList(head[0])
        self.conf = torch.nn.ModuleList(head[1])
        # print(self.loc)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        phase = kwargs['phase']
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            # TODO: different with lite this one should be change
            if k % 2 == 1:
                sources.append(x)

        assert len(self.transforms) == len(sources)
        upsize = (sources[0].size()[2], sources[0].size()[3])

        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        if phase == 'feature':
            return pyramids

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def _add_extras_(self, feature_layer, mbox, num_classes, version):
        extra_layers = []
        feature_transform_layers = []
        pyramid_feature_layers = []
        loc_layers = []
        conf_layers = []
        in_channels = None
        feature_transform_channel = int(feature_layer[0][1][-1] / 2)
        for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
            if version == 'T':
                if layer == 'S':
                    extra_layers += [self._conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1)]
                    in_channels = depth
                elif layer == '':
                    extra_layers += [self._conv_dw(in_channels, depth, stride=1, expand_ratio=1)]
                    in_channels = depth
                else:
                    in_channels = depth
            else:
                if layer == 'S':
                    extra_layers += [
                        torch.nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                        torch.nn.Conv2d(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
                    in_channels = depth
                elif layer == '':
                    extra_layers += [
                        torch.nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                        torch.nn.Conv2d(int(depth / 2), depth, kernel_size=3)]
                    in_channels = depth
                else:
                    in_channels = depth
            feature_transform_layers += [BasicConv(in_channels, feature_transform_channel, kernel_size=1, padding=0, bn=False, bias=True)]

        in_channels = len(feature_transform_layers) * feature_transform_channel
        for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
            if layer == 'S':
                pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=2, padding=1, bn=False, bias=True)]
                in_channels = depth
            elif layer == '':
                pad = (0, 1)[len(pyramid_feature_layers) == 0]
                pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=1, padding=pad, bn=False, bias=True)]
                in_channels = depth
            else:
                AssertionError('Undefined layer')
            loc_layers += [torch.nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
            conf_layers += [torch.nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
        return extra_layers, (feature_transform_layers, pyramid_feature_layers), (loc_layers, conf_layers)

    def _conv_dw(self, inp, oup, stride=1, padding=0, expand_ratio=1):
        return torch.nn.Sequential(
            # pw
            torch.nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # dw
            torch.nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio,
                            bias=False),
            torch.nn.BatchNorm2d(oup * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # pw-linear
            torch.nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        )


class BasicConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = torch.nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU(inplace=True) if relu else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = torch.nn.functional.upsample(x, size=up_size, mode='bilinear')
            # x = self.up_sample(x)
        return x
