'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: rfb.py
@time: 2021-09-25 14:49:02
@desc: 
'''
import torch
from lib.model.base import ModelBase
from lib.model.model_zoo import MODEL_ZOO


@MODEL_ZOO.register()
def rfb(cfg):
    return RFB(cfg)


class RFB(ModelBase):
    def __init__(self, cfg):
        super(RFB, self).__init__(cfg)
        self.base = torch.nn.ModuleList(self.backbone)

        if self.cfg['net']['lite'] == 'T':
            extras,norm, head = self._add_extras_lite_(self.feature_layer,
                                                       self.number_box,
                                                       self.num_classes,
                                                       self.cfg['net']['lite'])
            self.norm = BasicRFB_a_lite(self.feature_layer[1][0],
                                        self.feature_layer[1][0], stride=1, scale=1.0)
        else:
            extras,norm, head = self._add_extras_(self.feature_layer,
                                                  self.number_box,
                                                  self.num_classes,
                                                  self.cfg['net']['lite'])
            self.norm = torch.nn.ModuleList(norm)

        self.extras = torch.nn.ModuleList(extras)
        self.loc = torch.nn.ModuleList(head[0])
        self.conf = torch.nn.ModuleList(head[1])
        self.softmax = torch.nn.Softmax(dim=-1)

        self.feature_layer = self.feature_layer[0]
        self.indicator = 0
        for layer in self.feature_layer:
            if isinstance(layer, int):
                continue
            elif layer == '' or layer == 'S':
                break
            else:
                self.indicator += 1

    def forward(self, x, **kwargs):
        phase = kwargs['phase']
        sources, loc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if self.cfg['net']['lite'] == 'T':
                    if len(sources) == 0:
                        s = self.norm(x)
                        sources.append(s)
                    else:
                        sources.append(x)
                else:
                    idx = self.feature_layer.index(k)
                    if (len(sources)) == 0:
                        sources.append(self.norm[idx](x))
                    else:
                        x = self.norm[idx](x)
                        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or (k % 2 == 1 if self.cfg['net']['lite'] == 'F' else k % 2 == 0):
                sources.append(x)

        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
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
        loc_layers = []
        conf_layers = []
        norm_layers = []
        in_channels = None
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if version == 'T':
                pass
            else:
                if layer == 'RBF':
                    extra_layers += [BasicRFB(in_channels, depth, stride=2, scale=1.0, visual=2)]
                    in_channels = depth
                elif layer == 'S':
                    extra_layers += [
                        BasicConv(in_channels, int(depth / 2), kernel_size=1),
                        BasicConv(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
                    in_channels = depth
                elif layer == '':
                    extra_layers += [
                        BasicConv(in_channels, int(depth / 2), kernel_size=1),
                        BasicConv(int(depth / 2), depth, kernel_size=3)]
                    in_channels = depth
                else:
                    if len(norm_layers) == 0:
                        norm_layers += [BasicRFB_a(depth, depth, stride=1, scale=1.0)]
                    else:
                        norm_layers += [BasicRFB(depth, depth, scale=1.0, visual=2)]
                    in_channels = depth
            loc_layers += [torch.nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
            conf_layers += [torch.nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
        return extra_layers, norm_layers, (loc_layers, conf_layers)

    def _add_extras_lite_(self, feature_layer, mbox, num_classes, version):
        extra_layers = []
        loc_layers = []
        conf_layers = []
        in_channels = None
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if layer == 'RBF':
                extra_layers += [BasicRFB_lite(in_channels, depth, stride=2, scale=1.0)]
                in_channels = depth
            elif layer == 'S':
                extra_layers += [
                    BasicConv(in_channels, int(depth / 2), kernel_size=1),
                    BasicConv(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
                in_channels = depth
            elif layer == '':
                extra_layers += [
                    BasicConv(in_channels, int(depth / 2), kernel_size=1),
                    BasicConv(int(depth / 2), depth, kernel_size=3)]
                in_channels = depth
            else:
                in_channels = depth
            loc_layers += [torch.nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
            conf_layers += [torch.nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
        return extra_layers, [],  (loc_layers, conf_layers)


class BasicRFB_a_lite(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a_lite, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
                )
        self.branch1 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        out = out*self.scale + x
        out = self.relu(out)

        return out


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


class BasicSepConv(torch.nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = torch.nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
        )
        self.branch1 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
        )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0,x1),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(torch.nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
        )
        self.branch1 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = torch.nn.Sequential(
            BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
            BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut   = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu       = torch.nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out


class BasicRFB_lite(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_lite, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = torch.nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=stride, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1,x2),1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale + short
        out = self.relu(out)
        return out
