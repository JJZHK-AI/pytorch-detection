"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: dataset.py
@time: 2021-11-05 21:13:07
@desc: 
"""
import torch.utils.data
import torchvision as tv
import numpy as np
from lib.dataset.data_zoo import DATASET_ZOO
from jjzhk.dataset import TestData, VOCData, COCOData, DataSetBase
from jjzhk.config import DetectConfig
from lib.yolov2.argument import BaseTransform, Compose, ConvertFromInts, ToAbsoluteCoords
from lib.yolov2.argument import PhotometricDistort, Expand, RandomSampleCrop, RandomMirror
from lib.yolov2.argument import ToPercentCoords, Resize, Normalize
import os
from lib.utils.util import write_voc_results_file, do_python_eval, write_coco_results_file, do_detection_eval


@DATASET_ZOO.register()
def yolov2_voc(cfg, phase):
    return VOCDetection(cfg, phase)


@DATASET_ZOO.register()
def yolov2_coco(cfg, phase):
    return COCODetection(cfg, phase)


class YoloV2Detection(DataSetBase):
    def __init__(self, cfg: DetectConfig, phase):
        super(YoloV2Detection, self).__init__(cfg, phase)
        self.transform = [tv.transforms.ToTensor()]
        self.mean = (123, 117, 104)


    def get_item(self, index):
        img, target = self.dataset.find_item(index)
        info = self.dataset.get_item_info(index)
        target = np.asarray(target)

        if self.phase == 'train':
            self.image_size = self.cfg['train']['imagesize']
            target_transform = TransformTarget()
            target = target_transform(target, info['width'], info['height'])
            transform = SSDAugmentation(self.image_size[0])
            if self.transform is not None:
                img, boxes, labels = transform(img, target[:, :4], target[:, 4])
                img = img[:, :, (2, 1, 0)]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        elif self.phase == 'eval':
            self.image_size = self.cfg['eval']['imagesize']
            transform = BaseTransform(self.image_size[0])
            img = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        else:
            self.image_size = self.cfg['test']['imagesize']
            transform = BaseTransform(self.image_size[0])
            img = torch.from_numpy(transform(img)[0][:, :, ::-1].copy()).permute(2, 0, 1)
            target = None
        return img, target, info

    def collater(self, batch):
        targets = []
        imgs = []
        infos = []
        for sample in batch:
            imgs.append(sample[0])
            if sample[1] is not None:
                targets.append(torch.FloatTensor(sample[1]))
            infos.append(sample[2])
        return torch.stack(imgs, 0), targets, infos


class VOCDetection(YoloV2Detection):
    def __init__(self, cfg: DetectConfig, phase):
        super(VOCDetection, self).__init__(cfg, phase)

    def __init_dataset__(self):
        if self.phase == "train":
            return VOCData(self.cfg, "train")
        elif self.phase == "eval":
            return VOCData(self.cfg, "test")
        elif self.phase == "test":
            return TestData(cfg=self.cfg)
        else:
            raise Exception("phase must be train, eval, test")

    def evaluate_detections(self, boxes, output_dir, infos):
        write_voc_results_file(self.cfg, output_dir, boxes, infos)
        return do_python_eval(self.cfg, infos, output_dir)


class COCODetection(YoloV2Detection):
    def __init__(self, cfg: DetectConfig, phase):
        super(COCODetection, self).__init__(cfg, phase)

    def __init_dataset__(self):
        if self.phase == "train":
            return COCOData(self.cfg, "train")
        elif self.phase == "eval":
            return COCOData(self.cfg, "test")
        elif self.phase == "test":
            return TestData(cfg=self.cfg)
        else:
            raise Exception("phase must be train, eval, test")

    def evaluate_detections(self, boxes, output_dir, infos):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self.coco_name +
                                             '_results'))
        res_file += '.json'
        write_coco_results_file(self.cfg, boxes,
                                res_file, infos,
                                dict(zip([c['name'] for c in self.cats],self.dataset.coco.getCatIds())),
                                )
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            return do_detection_eval(self.cfg, self.dataset.annFile, res_file)


class SSDAugmentation(object):
    def __init__(self, size=416, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.mean_255 = (mean[0]*255, mean[1]*255, mean[2]*255)
        self.size = size
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean_255),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class TransformTarget(object):
    def __init__(self):
        pass

    def __call__(self, target, width, height):
        res = []
        boxes = target[:, 1:5]
        for j, t in enumerate(boxes):
            bndbox = []
            for i, pt in enumerate(t):
                cur_pt = pt / width if i % 2 == 0 else pt / height
                bndbox.append(cur_pt)
            bndbox.append(target[j][0])
            res += [bndbox]
        return np.asarray(res)

