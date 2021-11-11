"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: eval.py
@time: 2021-11-10 15:58:30
@desc: 
"""
from jjzhk.config import DetectConfig


class EvalObj:
    def __init__(self, config: DetectConfig, model):
        self.cfg = config
        self.model = model

    def calculateMAP(self, loader, output_path, **kwargs):
        pass