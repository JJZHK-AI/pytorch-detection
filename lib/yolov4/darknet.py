"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: darknet.py
@time: 2021-09-27 11:31:31
@desc: 
"""
from pathlib import Path
import numpy as np
from lib.yolov4.utils import scale_img, create_modules
import torch


class Darknet(torch.nn.Module):
    def __init__(self, cfg):
        super(Darknet, self).__init__()
        self.cfg = cfg
        self.module_defs = self.cfg['backbone']
        self.module_list, self.routs = \
            create_modules(self.module_defs, cfg)
        self.module_list = torch.nn.ModuleList(self.module_list)
        # self.module_list, self.routs = create_modules(self.module_defs, 640, self.cfg['backbone'])

    def load_darknet_weights(self, weights, cutoff=-1):
        # Parses and loads the weights stored in 'weights'

        # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
        file = Path(weights).name
        if file == 'darknet53.conv.74':
            cutoff = 75
        elif file == 'yolov3-tiny.conv.15':
            cutoff = 15

        # Read weights file
        with open(weights, 'rb') as f:
            # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
            self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

        ptr = 0
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv = module[0]
                if mdef['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn = module[1]
                    nb = bn.bias.numel()  # number of biases
                    # Bias
                    bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                    ptr += nb
                    # Weight
                    bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                    ptr += nb
                    # Running Mean
                    bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                    ptr += nb
                    # Running Var
                    bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                    ptr += nb
                else:
                    # Load conv. bias
                    nb = conv.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += nb
                # Load conv. weights
                nw = conv.weight.numel()  # number of weights
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
                ptr += nw

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate(
                    (x,scale_img(x.flip(3), s[0], same_shape=False),scale_img(x, s[1], same_shape=False),)):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           scale_img(x, s[1]),  # scale
                           ), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            #print(name)
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ScaleSpatial']:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            elif name == 'JDELayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                #print(module)
                #print(x.shape)
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def summary(self):
        print(self.model_summary)
        for index, layer in enumerate(self.module_list):
            print("%d %s" % (index, layer))

