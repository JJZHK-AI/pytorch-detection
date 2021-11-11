'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: eval.py
@time: 2020-07-17 10:14:33
@desc: 
'''
from jjzhk.config import DetectConfig
from lib.utils.eval import EvalObj


class SSDEval(EvalObj):
    def __init__(self, config: DetectConfig, model):
        super(SSDEval, self).__init__(config, model)

    def eval_boxes(self, batch, **kwargs) -> tuple:
        images, targets, info = batch['img'], batch['annot'], batch['info']
        detector = kwargs.get('detector')
        detections = self.model.get_detections(images, detector=detector)

        return detections, info[0]


