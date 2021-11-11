'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: eval.py
@time: 2020-07-17 10:14:33
@desc: 
'''
from lib.utils.eval import EvalObj
from jjzhk.config import DetectConfig
from jjzhk.progressbar import ProgressBar


class SSDEval(EvalObj):
    def __init__(self, config: DetectConfig, model):
        super(SSDEval, self).__init__(config, model)

    def calculateMAP(self, loader, output_path, **kwargs):
        all_boxes = [[[] for _ in range(len(loader))]
                     for _ in range(len(self.cfg.class_keys()) + 1)]
        infos = []
        bar = ProgressBar(1, len(loader), "")
        for i, sampler in enumerate(loader):
            images, targets, info = sampler['img'], sampler['annot'], sampler['info']
            detector = kwargs.get('detector')
            detections = self.model.get_detections(images, detector=detector)

            # print(info)
            image_eval_boxes = self.model.get_eval_predictions(info, detections)

            for j, box in enumerate(image_eval_boxes):
                all_boxes[j][i] = box

            infos.append(info)
            bar.show(1)

        print("calculating mAP...")
        return loader.dataset.evaluate_detections(all_boxes, output_path, infos)


