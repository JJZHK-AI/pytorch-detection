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
import numpy as np
import torchvision as tv
from lib.dataset.data_zoo import DATASET_ZOO
from jjzhk.dataset import TestData, VOCData, COCOData, DataSetBase
from jjzhk.config import DetectConfig
from lib.yolov1.arguments import random_flip, randomBlur, randomCrop, randomScale, RandomBrightness
from lib.yolov1.arguments import RandomHue, RandomSaturation, randomShift, BGR2RGB, subMean
import cv2
import os
from lib.utils.util import write_voc_results_file, do_python_eval, write_coco_results_file, do_detection_eval


@DATASET_ZOO.register()
def yolov1_voc(cfg, phase):
    return VOCDetection(cfg, phase)


@DATASET_ZOO.register()
def yolov1_coco(cfg, phase):
    return COCODetection(cfg, phase)


class YoloV1Detection(DataSetBase):
    def __init__(self, cfg: DetectConfig, phase):
        super(YoloV1Detection, self).__init__(cfg, phase)
        self.transform = [tv.transforms.ToTensor()]
        self.mean = tuple(self.cfg['base']['means'])
        self.image_size = self.cfg['net']['imagesize'][0]

    def get_item(self, index):
        # target [[label, left, top, right, bottom]]
        img, target = self.dataset.find_item(index)

        info = self.dataset.get_item_info(index)
        target = np.asarray(target)
        if self.phase == 'train':
            labels = torch.Tensor(target[:, 0]) + 1  # 取出来的label是0开始的，所以需要加1
            boxes = torch.Tensor(target[:, 1:5])
            img, boxes = random_flip(img, boxes)
            img, boxes = randomScale(img, boxes)
            img = randomBlur(img)
            img = RandomBrightness(img)
            img = RandomHue(img)
            img = RandomSaturation(img)
            img, boxes, labels = randomShift(img, boxes, labels)
            img, boxes, labels = randomCrop(img, boxes, labels)

            h, w, _ = img.shape
            boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
            img = BGR2RGB(img)  # because pytorch pretrained model use RGB
            img = subMean(img, self.mean)  # 减去均值
            img = cv2.resize(img, (self.image_size, self.image_size))
            target = self.encoder(boxes, labels).cpu().numpy()  # 7x7x30
            for t in self.transform:
                img = t(img)
        elif self.phase == 'eval':
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = BGR2RGB(img)
            img = subMean(img, self.mean)  # 减去均值
            for t in self.transform:
                img = t(img)
        else:
            h, w, _ = img.shape
            img = cv2.resize(img, tuple(self.cfg['net']['imagesize']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean = self.mean  # RGB
            img = img - np.array(mean, dtype=np.float32)

            transform = tv.transforms.Compose([tv.transforms.ToTensor(), ])
            img = transform(img)

        return img, target, info

    # def __len__(self):
    #     return 1

    def encoder(self, boxes, labels):
        """
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        """
        grid_num = self.cfg['net']['cell_number']
        target = torch.zeros((grid_num, grid_num, 2 * 5 + self.cfg['dataset']['classno']))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def collater(self, batch):
        targets = []
        imgs = []
        infos = []
        for sample in batch:
            imgs.append(sample[0])
            if sample[1] is not None:
                targets.append(sample[1])
            infos.append(sample[2])
        return np.stack(imgs, 0), np.stack(targets, 0), infos


class VOCDetection(YoloV1Detection):
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


class COCODetection(YoloV1Detection):
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
