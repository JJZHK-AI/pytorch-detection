'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: data_zoo.py
@time: 2020-06-16 10:44:58
@desc: 
'''
from jjzhk.register import Registry

DATASET_ZOO = Registry("DATASET")


def get_dataset(cfg, phase):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = DATASET_ZOO.get(cfg['dataset']['name'])(cfg, phase)
    return model