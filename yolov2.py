"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov2.py
@time: 2021-11-12 14:13:42
@desc: 
"""
import lib.yolov2 as y
from jjzhk.config import DetectConfig
import os

config = DetectConfig("cfg")
config.load_file_list([
        "%s.cfg" % "voc",
        os.path.join("%d" % 448, "%s" % 'coco', "%s_%s.cfg" % ("yolov2", "darknet19"))])
config.load_backbone_file(os.path.join("backbone", "yolov2_%s.cfg" % "darknet19"))
net = y.YOLOV2D19(config)
print(net)

