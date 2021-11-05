"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: data.py
@time: 2021-09-29 16:49:47
@desc: 
"""
import numpy as np
import cv2
import os
import torch
import torch.utils.data
from jjzhk.dataset import TestData, VOCData, COCOData
from jjzhk.dataset import DataSetBase
from jjzhk.config import DetectConfig
from lib.dataset.data_zoo import DATASET_ZOO
from tqdm import tqdm
from PIL import Image
from lib.ssd.utils import get_hash, exif_size
from pathlib import Path


@DATASET_ZOO.register()
def yolov4_coco(cfg, phase):
    return COCODataSet(cfg, phase)


class YoloV4DataSet(DataSetBase):
    def __init__(self, cfg: DetectConfig, phase):
        super(YoloV4DataSet, self).__init__(cfg, phase)

        if phase == 'eval':
            self.img_files = [os.path.join(self.cfg['dataset']['root'], 'val2017', x['path']) for x in
                              self.dataset._img_list]
            self.label_files = [x.replace('jpg', 'txt') for x in self.img_files]
            self.rect = True
            self.augment = False
            cache_path = os.path.join(cfg['dataset']['root'], 'val2017.cache3')  # cached labels
            if os.path.isfile(cache_path):
                cache = torch.load(cache_path)  # load
                if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                    cache = self.cache_labels(cache_path)  # re-cache
            else:
                cache = self.cache_labels(cache_path)  # cache

            cache.pop('hash')  # remove hash
            self.labels, shapes = zip(*cache.values())
            self.shapes = np.array(shapes, dtype=np.float64)
            self.batch_shapes = self._get_batch_shapes_()

    def get_item(self, index):
        image, target = self.dataset.find_item(index)
        info = self.dataset.get_item_info(index)
        img = image

        if self.phase == 'train':
            pass
        elif self.phase == 'eval':
            h0, w0 = img.shape[:2]
            r = 640 / max(h0, w0)
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            h, w = img.shape[:2]  # img, hw_original, hw_resized
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels_out = torch.zeros((0, 6))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            return torch.from_numpy(img), labels_out, info, shapes

        elif self.phase == 'test':
            self.auto_size = 32
            height, width, _ = image.shape
            img = letterbox(image, new_shape=640, auto_size=self.auto_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            return img, info

    #region private
    def _get_batch_shapes_(self):
        n = len(self)
        bi = np.floor(np.arange(n) / 32).astype(np.int)
        nb = bi[-1] + 1
        self.batch = bi
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.img_files = [self.img_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        return np.ceil(np.array(shapes) * 640 / 64 + 0.5).astype(np.int) * 64

    def cache_labels(self, path='labels.cache3'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict

        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                im = Image.open(img)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x
    #endregion


class COCODataSet(YoloV4DataSet):
    def __init__(self,cfg:DetectConfig, phase):
        super(COCODataSet, self).__init__(cfg, phase)
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


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
