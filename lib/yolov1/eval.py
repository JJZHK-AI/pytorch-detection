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
import torch
from jjzhk.device import device


class YOLOV1Eval(EvalObj):
    def __init__(self, config: DetectConfig, model):
        super(YOLOV1Eval, self).__init__(config, model)

    def eval_boxes(self, batch, **kwargs) -> tuple:
        images, info = batch[0], batch[2]
        images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
        detections = self.model(images)

        return detections, info[0]