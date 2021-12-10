"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: darknet.py
@time: 2021-09-27 11:31:31
@desc: 
"""
from jjzhk.device import device
from jjzhk.config import DetectConfig
from lib.yolov4.utils import scale_img, create_modules
import torch
from lib.backbone.backbone_layer import get_backbone
from lib.yolov4.utils import non_max_suppression, scale_coords
from lib.model.base import ModelBase


class Darknet(ModelBase):
    def __init__(self, cfg: DetectConfig):
        super(Darknet, self).__init__(cfg)
        self.module_defs = self.cfg['backbone']
        self.module_list, self.routs = \
            create_modules(self.module_defs, cfg)
        self.module_list = torch.nn.ModuleList(self.module_list)

    def forward(self, x, **kwargs):
        augment = kwargs.get('augment')
        verbose = kwargs.get('verbose')
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
        for index, layer in enumerate(self.module_list):
            print("%d %s" % (index, layer))

    def get_eval_predictions(self, sampler, **kwargs):
        pass
        # images, info = sampler[0], sampler[2]
        # images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
        # bboxes, scores, cls_inds = self(images)
        #
        # result = []
        # w, h = info[0]['width'], info[0]['height']
        #
        # for i, box in enumerate(bboxes):
        #     prob = scores[i]
        #     prob = float(prob)
        #     if prob >= self.cfg['base']['conf_threshold']:
        #         x1 = int(box[0] * w)
        #         x2 = int(box[2] * w)
        #         y1 = int(box[1] * h)
        #         y2 = int(box[3] * h)
        #
        #         result.append([(x1, y1), (x2, y2), int(cls_inds[i]), self.cfg.classname(int(cls_inds[i])), prob])
        #
        # re_boxes = [[] for _ in range(len(self.cfg.class_keys()) + 1)]
        # for (x1, y1), (x2, y2), class_id, class_name, prob in result: #image_id is actually image_path
        #     re_boxes[class_id+1].append([x1, y1, x2, y2, prob])
        #
        # return re_boxes

    def get_test_predict(self, image, info, **kwargs):
        img = image.to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self(img, augment=False)[0]
        pred = non_max_suppression(pred, self.cfg['base']['conf_threshold'],
                                     self.cfg['base']['iou_threshold'],
                                     classes=None, agnostic=False)

        result = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (info['height'].item(), info['width'].item(), 3)).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()

                for *xyxy, conf, cls in det:
                    result.append(
                        [
                            (xyxy[0].item(), xyxy[1].item()),
                            (xyxy[2].item(), xyxy[3].item()),
                            self.cfg.classname(int(cls)),
                            "", conf.item()
                        ]
                    )

        return result

    def load_init_weights(self, weights):
        # for k in weights.keys():
        #     print(k, weights[k].shape)
        self.load_state_dict(weights)



