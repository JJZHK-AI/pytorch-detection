"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: eval.py
@time: 2021-11-10 15:58:30
@desc: 
"""
import os
import numpy as np
from jjzhk.config import DetectConfig
from jjzhk.progressbar import ProgressBar


class EvalObj:
    def __init__(self, config: DetectConfig, model):
        self.cfg = config
        self.model = model

    def eval_boxes(self, batch, **kwargs) -> tuple:
        pass

    def calculateMAP(self, loader, output_path, **kwargs):
        all_boxes = [[[] for _ in range(len(loader))]
                     for _ in range(len(self.cfg.class_keys()) + 1)]
        infos = []
        bar = ProgressBar(1, len(loader), "")
        for i, sampler in enumerate(loader):
            detections, info = self.eval_boxes(sampler, **kwargs)
            # print(info)
            image_eval_boxes = self.model.get_eval_predictions(info, detections)

            for j, box in enumerate(image_eval_boxes):
                all_boxes[j][i] = box

            infos.append(info)
            bar.show(1)

        print("calculating mAP...")
        return loader.dataset.evaluate_detections(all_boxes, output_path, infos)