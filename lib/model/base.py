'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: base.py
@time: 2021-09-25 14:51:24
@desc: 
'''
import torch
from jjzhk.device import device
import numpy as np
from lib.backbone.util import create_modules


class ModelBase(torch.nn.Module):
    def __init__(self, cfg, callback=None):
        super(ModelBase, self).__init__()
        self.cfg = cfg
        self.num_classes = self.cfg['dataset']['classno']
        self.feature_layer = self.cfg['net']['features']
        self.number_box = [2 * len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for
                           aspect_ratios in self.cfg['net']['aspect_ratio']]

        self.module_defs = self.cfg['backbone']
        self.base, _, _ =  create_modules(self.module_defs, self.cfg, callback)
        self.base = torch.nn.ModuleList(self.base)

    def forward(self, x, **kwargs):
        pass

    def load_init_weights(self, weights):
        if 'state_dict' in weights.keys():
            self.load_state_dict(weights['state_dict'])
        else:
            self.load_state_dict(weights)

    def get_detections(self,image, **kwargs):
        with torch.no_grad():
            image = torch.autograd.Variable(torch.FloatTensor(image))

        image = image.to(device)
        pred = self.forward(image, phase='eval')
        detections = kwargs["detector"].forward(pred)

        return detections

    def get_eval_predictions(self,info, detections):
        w, h = info['width'], info['height']

        re_boxes = [[] for _ in range(len(self.cfg.class_keys()) + 1)]
        scale = [w, h, w, h]
        for j in range(1, detections.size(1)):
            cls_dets = list()
            for det in detections[0][j]:
                if det[0] > 0:
                    d = det.cpu().numpy()
                    scores, boxes = d[0], d[1:]
                    boxes *= scale
                    boxes = np.append(boxes, scores)
                    cls_dets.append(boxes)
            re_boxes[j] = cls_dets
        return re_boxes

    def get_predict(self, image, info, **kwargs):
        detections = self.get_detections(image, **kwargs)

        result_box = []

        for p, detection in enumerate(detections):
            image_id = info[p]["img_id"]
            height, width = info[p]["height"], info[p]["width"]
            scale = torch.Tensor([width, height, width, height])
            result = []

            for i in range(detections.size(1)):
                j = 0
                while detections[p, i, j, 0] >= float(self.cfg['base']['iou_threshold']):
                    score = detections[p, i, j, 0]
                    label_name = self.cfg.classname(i - 1)
                    pt = (detections[p, i, j, 1:] * scale).cpu().numpy()
                    result.append(
                        [
                            (pt[0], pt[1]),
                            (pt[2], pt[3]),
                            label_name,
                            image_id,
                            score
                        ]
                    )
                    j += 1
            result_box.append(result)
        return result_box