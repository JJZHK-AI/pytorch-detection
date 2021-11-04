"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: layer_zoo.py
@time: 2021-09-16 17:38:51
@desc: 
"""
from jjzhk.register import Registry


LAYER_ZOO = Registry('LAYER')


def get_layer(layercfg, **kwargs):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = LAYER_ZOO.get(layercfg['type'])(layercfg, **kwargs)
    return model
