"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: test.py
@time: 2021-11-09 13:52:55
@desc: 
"""
import torch
import torchvision as tv
from lib.yolov1.model import ResNet50


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = [
            torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(192, 128, kernel_size=(1, 1)),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),

            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(512, 512, kernel_size=(1, 1)),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1)),

            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1))
        ]

    def forward(self, x):
        input = x
        for layer in self.layers:
            input = layer(input)

        return input


image = torch.randn((1, 3, 448, 448))
# model = MyModel()
# output = model(image)
# print(output.shape)
#
# vgg = tv.models.vgg16_bn(pretrained=True)
# features = vgg.features
# avgpool = vgg.avgpool
# output = features(image)
# output = avgpool(output)
# print(output.shape)
#
resnet = tv.models.resnet50(pretrained=True)
print(resnet)
output = resnet.conv1(image)
print(output.shape)
output = resnet.bn1(output)
print(output.shape)
output = resnet.relu(output)
print(output.shape)
output = resnet.maxpool(output)
print(output.shape)
output = resnet.layer1(output)
print(output.shape)
output = resnet.layer2(output)
print(output.shape)
output = resnet.layer3(output)
print(output.shape)
output = resnet.layer4(output)
print(output.shape)
# output = resnet(output)
# print(output.shape)
# print('--------------')
# model = ResNet50()
# output = model(image)
# print(output.shape)