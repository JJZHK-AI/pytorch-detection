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
import torchvision as tv

model = tv.models.vgg19_bn(pretrained=True)
image = torch.zeros(1, 3, 448, 448)
output = model.features(image)
print(output.shape) # 1, 512, 14, 14
output = model.avgpool(output)
print(output.shape) # 1, 512, 7, 7






