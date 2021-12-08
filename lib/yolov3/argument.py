"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: argument.py
@time: 2021-11-17 15:31:06
@desc: 
"""
import numpy as np
import cv2


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x