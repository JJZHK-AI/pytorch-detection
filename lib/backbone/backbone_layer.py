"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: backbone_layer.py
@time: 2021-11-17 18:32:52
@desc: 
"""
from jjzhk.register import Registry

BACKBONE_ZOO = Registry("DATASET")


def get_backbone(cfg):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_ZOO.get(cfg['net']['backbone'])(cfg)
    return model
