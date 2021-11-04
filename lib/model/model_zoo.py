'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: model_zoo.py
@time: 2021-09-25 14:17:52
@desc: 
'''
from jjzhk.register import Registry


MODEL_ZOO = Registry('MODEL')


def get_model(name, cfg):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = MODEL_ZOO.get(name)(cfg)
    return model