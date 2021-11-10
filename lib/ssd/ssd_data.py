'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: SSD.py
@time: 2020-06-16 10:41:34
@desc: 
'''

import torch
import numpy as np
import cv2
import random
import math
import json
import os

from lib.utils.util import write_voc_results_file, do_python_eval
from jjzhk.config import DetectConfig
from lib.dataset.data_zoo import DATASET_ZOO
from jjzhk.dataset import TestData, VOCData, COCOData, DataSetBase
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


@DATASET_ZOO.register()
def voc(cfg, phase):
    return VOCDetection(cfg, phase)


@DATASET_ZOO.register()
def coco(cfg, phase):
    return COCODetection(cfg, phase)


class SSDDetection(DataSetBase):
    def __init__(self, cfg: DetectConfig, phase):
        super(SSDDetection, self).__init__(cfg, phase)

        self.preproc = self._init_preproc_()

    # def __len__(self):
    #     return 1

    def get_item(self, index):
        # index = self.dataset.__getIndexByImageId__("005144")
        img, target = self.dataset.find_item(index)
        info = self.dataset.get_item_info(index)
        height, width, _ = img.shape

        if self.phase != 'test':
            target = np.asarray(target, dtype=np.float32)
            boxes = target[:, 1:5] - 1
            labels = target[:, 0] + 1
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img.numpy(), target, info

    def _init_preproc_(self):
        if self.phase == "train":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], float(self.cfg['dataset']['prob']))
        elif self.phase == "eval":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], -1)
        elif self.phase == "test":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], -2)
        else:
            raise Exception("phase must be train, eval, test")

    def collater(self, batch):
        targets = []
        imgs = []
        infos = []
        for sample in batch:
            imgs.append(sample[0])
            if sample[1] is not None:
                targets.append(sample[1])
            infos.append(sample[2])
        return {'img': np.stack(imgs, 0), 'annot': targets, 'info': infos}


class VOCDetection(SSDDetection):
    def __init__(self,cfg:DetectConfig, phase):
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


class COCODetection(SSDDetection):
    def __init__(self, cfg, phase):
        super(COCODetection, self).__init__(cfg, phase)
        self.coco_name = "eval2017"
        if phase == 'eval':
            self.cats = self.dataset.coco.loadCats(self.dataset.coco.getCatIds())

        self.classes = tuple(['__background__'] + self.cfg.keys())

    def __init_dataset__(self):
        if self.phase == "train":
            return COCOData(self.cfg, "train")
        elif self.phase == "eval":
            return COCOData(self.cfg, "val")
        elif self.phase == "test":
            return TestData(cfg=self.cfg)
        else:
            raise Exception("phase must be train, eval, test")

    def get_item(self, index):
        img, target = self.dataset.get_item(index)
        info = self.dataset.get_item_info(index)
        height, width, _ = img.shape
        #
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        #
        return img.numpy(), target, info

    def _init_preproc_(self):
        if self.phase == "train":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], float(self.cfg['dataset']['prob']))
        elif self.phase == "eval":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], -1)
        elif self.phase == "test":
            return preproc(self.cfg['net']['imagesize'], self.cfg['dataset']['means'], -2)
        else:
            raise Exception("phase must be train, eval, test")

    def evaluate_detections(self, boxes, output_dir, infos):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self.coco_name +
                                             '_results'))
        res_file += '.json'
        self.write_coco_results_file(boxes, res_file, infos)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            return self.do_detection_eval(res_file)

    def write_coco_results_file(self, all_boxes, res_file, infos):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []

        class_to_coco_cat_id = dict(zip([c['name'] for c in self.cats],
                                        self.dataset.coco.getCatIds()))

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             len(self.classes) - 1))
            coco_cat_id = class_to_coco_cat_id[cls]
            results.extend(self.coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id, infos))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
        # res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def coco_results_one_category(self, boxes, cat_id, infos):
        results = []
        for im_ind, info in enumerate(infos):
            dets = np.array(boxes[im_ind]).astype(np.float)
            if list(dets) == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': info['img_id'],
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def do_detection_eval(self, res_file):
        coco = COCO(self.dataset.annFile)
        ann_type = 'bbox'
        coco_dt = coco.loadRes(res_file)
        coco_eval = COCOeval(coco, coco_dt, ann_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        return self._print_detection_eval_metrics(coco_eval)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])

        mAP = ap_default
        infos = {}
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])

            infos[cls] = ap
        return mAP, infos


class preproc(object):
    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p # -1:eval; -2:test
        self.epoch = 0

    def __call__(self, image, targets=None):
        # some bugs
        if self.p == -2: # test
            targets = np.zeros((1,5))
            targets[0] = image.shape[0]
            targets[0] = image.shape[1]
            image = preproc_for_test(image, self.resize, self.means) # resize + 去均值化
            return torch.from_numpy(image), targets

        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            image = preproc_for_test(image, self.resize, self.means) # some ground truth in coco do not have bounding box! weird!
            return torch.from_numpy(image), targets
        if self.p == -1: # eval
            height, width, _ = image.shape
            boxes[:, 0::2] /= width # 归一化
            boxes[:, 1::2] /= height
            labels = np.expand_dims(labels,1)
            targets = np.hstack((boxes,labels))
            image = preproc_for_test(image, self.resize, self.means) # resize + 去均值化
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o # 去均值化
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        image_t, boxes, labels = _crop(image, boxes, labels)

        image_t = _distort(image_t)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t)==0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        return torch.from_numpy(image_t), targets_t

    def add_writer(self, writer, epoch=None):
        self.writer = writer
        self.epoch = epoch if epoch is not None else self.epoch + 1

    def release_writer(self):
        self.writer = None


'''
resize + 去均值化
'''
def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize[0], insize[1]),interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)


def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t,labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1,4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def matrix_iou(a,b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)
