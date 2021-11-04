"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: layers.py
@time: 2021-10-27 13:13:31
@desc: 
"""
import torch
import torch.nn.functional
from lib.backbone.conv2d import Conv2d
from lib.backbone.layer_zoo import  LAYER_ZOO
from lib.backbone.util import layer_to_config
import math
from lib.backbone.util import get_layer


def yolov4_create_modules(mdef, cfg, **kwargs):
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

    if mdef['type'] == 'route':
        layers = mdef['layers']
        filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
        routs.extend([index + l if l < 0 else l for l in layers])
        l = get_layer(mdef, layers=layers)
    elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
        layers = mdef['from']
        filters = inFilters
        routs.extend([index + l if l < 0 else l for l in layers])
        l = get_layer(mdef, layers=layers, weight='weights_type' in mdef)
    elif mdef['type'] == 'upsample':
        l = torch.nn.Upsample(scale_factor=mdef['stride'])
    elif mdef['type'] == 'yolo':
        yolo_index = mdefsummary['yolo']
        stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
        if any(x in mdef for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
            stride = [32, 16, 8]
        layers = mdef['from'] if 'from' in mdef else []
        modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                            nc=mdef['classes'],  # number of classes
                            img_size=img_size,  # (416, 416)
                            yolo_index=yolo_index,  # 0, 1, 2...
                            layers=layers,  # output layers
                            stride=stride[yolo_index])

        # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
        try:
            j = layers[yolo_index] if 'from' in mdef else -1
            bias_ = module_list[j][0].bias  # shape(255,)
            bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
            # bias[:, 4] += -4.5  # obj
            bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
            bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
            module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
        except:
            print('WARNING: smart bias initialization failure.')
        l = modules
    else:
        print('Warning: Unrecognized Layer Type: ' + mdef['type'])
    return l, filters


@LAYER_ZOO.register(version=1)
def convolutional(layer, **kwargs):
    return Yolov4Conv2d(layer, **kwargs).layers()


@LAYER_ZOO.register()
def route(layer, **kwargs):
    layers = kwargs['layers']

    return FeatureConcat(layers=layers)


@LAYER_ZOO.register()
def shortcut(layer, **kwargs):
    layers = kwargs['layers']
    weight = kwargs['weight']
    return WeightedFeatureFusion(layers=layers, weight=weight)


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()


class FeatureConcat(torch.nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]

    def __str__(self):
        return "FeatureConcat(%s)" % self.layers


class WeightedFeatureFusion(torch.nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x

    def __str__(self):
        return "WeightedFeatureFusion(layers=%s, weight=%s)" % (self.layers, self.weight)


class Yolov4Conv2d(Conv2d):
    def __init__(self, layer_config, **kwargs):
        super(Yolov4Conv2d, self).__init__(layer_config, **kwargs)

    def transfer_config(self, **kwargs):
        list = []
        bn = self._layer_config_['batch_normalize']
        filters = self._layer_config_['filters']
        k = self._layer_config_['size']  # kernel size
        stride = self._layer_config_['stride'] if 'stride' in self._layer_config_ else (
            self._layer_config_['stride_y'], self._layer_config_['stride_x'])
        in_filter = self._layer_config_["inFilters"]
        layer_index = self._layer_config_['index']
        routs = kwargs['routs']

        if isinstance(k, int):  # single-size conv
            list.append(layer_to_config("Conv2d", torch.nn.Conv2d(in_channels=in_filter, out_channels=filters,
                                                                  kernel_size=(k, k), stride=stride,
                                                                  padding=self._layer_config_[
                                                                      'pad'] if 'pad' in self._layer_config_ else 0,
                                                                  groups=self._layer_config_[
                                                                      'groups'] if 'groups' in self._layer_config_ else 1,
                                                                  dilation=self._layer_config_[
                                                                      'dilation'] if 'dilation' in self._layer_config_ else 1,
                                                                  bias=not bn)))

        if bn:
            list.append(layer_to_config("BatchNorm2d", torch.nn.BatchNorm2d(filters, eps=0.0001, momentum=0.03)))
        else:
            routs.append(layer_index)  # detection output (goes into yolo layer)

        if self._layer_config_['activation'] == 'leaky':
            list.append(layer_to_config("activation", torch.nn.LeakyReLU(0.1, inplace=True)))
        elif self._layer_config_['activation'] == 'relu':
            list.append(layer_to_config("activation", torch.nn.ReLU(inplace=True)))
        elif self._layer_config_['activation'] == 'swish':
            list.append(layer_to_config('activation', Swish()))
        elif self._layer_config_['activation'] == 'mish':
            list.append(layer_to_config('activation', Mish()))

        return list


class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def __str__(self):
        return "YOLOLayer(anchors=%s, nc=%d, yolo_index=%d, layers=%s, stride=%d)" % \
            (self.anchors, self.nc, self.index, self.layers, self.stride)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         torch.nn.functional.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            #io = p.clone()  # inference output
            #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            #io[..., :4] *= self.stride
            #torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]
