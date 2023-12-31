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
from lib.backbone.backbone_layer import get_backbone
from jjzhk.config import DetectConfig


class ModelBase(torch.nn.Module):
    def __init__(self, cfg: DetectConfig):
        super(ModelBase, self).__init__()
        self.cfg = cfg

    def forward(self, x, **kwargs):
        pass

    def get_detections(self, image, **kwargs):
        pass

    def load_backbone_weights(self, weights):
        pass

    def load_init_weights(self, weights):
        pass

    def get_eval_predictions(self, sampler, **kwargs):
        pass

    def get_test_predict(self, image, info, **kwargs):
        pass

class DetectionModel(ModelBase):
    def __init__(self, cfg: DetectConfig):
        super(DetectionModel, self).__init__(cfg)
        self.num_classes = self.cfg['dataset']['classno']
        self.feature_layer = self.cfg['net']['features']
        self.number_box = [2 * len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for
                           aspect_ratios in self.cfg['net']['aspect_ratio']]

        self.base = torch.nn.ModuleList(get_backbone(self.cfg))

    def load_init_weights(self, weights):
        if 'state_dict' in weights.keys():
            self.load_state_dict(weights['state_dict'])
        else:
            self.load_state_dict(weights)

    def load_backbone_weights(self, weights):
        self.base.load_state_dict(weights)

    def get_detections(self,image, **kwargs):
        with torch.no_grad():
            image = torch.autograd.Variable(torch.FloatTensor(image))

        image = image.to(device)
        pred = self.forward(image, phase='eval')
        detections = kwargs["detector"].forward(pred)

        return detections

    def get_eval_predictions(self,sampler, **kwargs):
        images, targets, info = sampler['img'], sampler['annot'], sampler['info']
        detector = kwargs.get('detector')
        detections = self.model.get_detections(images, detector=detector)
        info = info[0]

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

    def get_test_predict(self, image, info, **kwargs):
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