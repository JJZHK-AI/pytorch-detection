"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: train.py
@time: 2021-11-05 20:56:41
@desc:
"""
import torch
from jjzhk.config import DetectConfig
import os
from jjzhk.drawseg import BaseDrawSeg
config = DetectConfig("cfg")
config.load_file_list([
    "%s.cfg" % "coco"
])
image_path = os.path.join(config['dataset']['test_root'], "Images")
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom

draw = BaseDrawSeg(cfg=config, output=os.path.join("output"))
list = sorted(os.listdir(image_path))
for file in list:
    if file.endswith(".jpg"):
        print(os.path.join(image_path, file))
        results = model(os.path.join(image_path, file))
        bbox = []
        for box in results.pred[0].numpy():
            prob = box[4]
            if prob > 0.5:
                bbox.append([
                    (box[0], box[1]), (box[2], box[3]),
                    config.classname((int)(box[5])),
                                "",prob
                ])
                print(config.classname((int)(box[5])))

        image = draw.draw_image(param={
                        "Image": os.path.join(image_path, file),
                        "Boxes": bbox,
                        "ImageName": file.split('.')[0]
                    }, draw_type=0)





