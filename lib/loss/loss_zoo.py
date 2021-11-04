'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: loss_zoo.py
@time: 2020-06-16 10:30:26
@desc: 
'''
from jjzhk.config import DetectConfig
from jjzhk.register import Registry

LOSS_ZOO = Registry("LOSS")


def get_loss(loss_name, cfg:DetectConfig, **kwargs):
    model = LOSS_ZOO.get(loss_name)(cfg, **kwargs)
    return model
