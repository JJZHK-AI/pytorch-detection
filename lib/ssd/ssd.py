'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: ssd.py
@time: 2020-06-16 09:57:22
@desc: 
'''
import torch
from lib.model.base import ModelBase
from lib.model.model_zoo import MODEL_ZOO


@MODEL_ZOO.register()
def ssd(cfg):
    return SSD(cfg)


class SSD(ModelBase):
    def __init__(self, cfg):
        super(SSD, self).__init__(cfg)
        extras, head = self._add_extras_(self.feature_layer,
                                         self.number_box,
                                         self.num_classes,
                                         self.cfg['net']['lite'])

        self.norm = L2Norm(self.feature_layer[1][0], 20)
        self.extras = torch.nn.ModuleList(extras)
        self.loc = torch.nn.ModuleList(head[0])
        self.conf = torch.nn.ModuleList(head[1])
        self.softmax = torch.nn.Softmax(dim=-1)

        self.feature_layer = self.feature_layer[0]

    def forward(self, x, **kwargs):
        phase = kwargs['phase']
        sources, loc, conf = [list() for _ in range(3)]
        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer: # 当K在feature_layer内部的时候，说明该加入图像金字塔了
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if self.cfg['net']['lite'] == 'T' or k % 2 == 1:
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
                loc.view(loc.size(0), -1, 4),                   # loc preds
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
        in_channels = None
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if version == 'T':
                if layer == 'S':
                    extra_layers += [self._conv_dw_(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
                    in_channels = depth
                elif layer == '':
                    extra_layers += [self._conv_dw_(in_channels, depth, stride=1, expand_ratio=1) ]
                    in_channels = depth
                else:
                    in_channels = depth
            else:
                if layer == 'S':
                    extra_layers += [
                        torch.nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                        torch.nn.Conv2d(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
                    in_channels = depth
                elif layer == '':
                    extra_layers += [
                        torch.nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                        torch.nn.Conv2d(int(depth/2), depth, kernel_size=3)  ]
                    in_channels = depth
                else:
                    in_channels = depth

            loc_layers  += [torch.nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
            conf_layers += [torch.nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
        return extra_layers, (loc_layers, conf_layers)

    def _conv_dw_(self, inp, oup, stride=1, padding=0, expand_ratio=1):
        return torch.nn.Sequential(
            # pw
            torch.nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # dw
            torch.nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
            torch.nn.BatchNorm2d(oup * expand_ratio),
            torch.nn.ReLU6(inplace=True),
            # pw-linear
            torch.nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        )


class L2Norm(torch.nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
