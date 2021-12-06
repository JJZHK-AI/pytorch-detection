"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: model.py
@time: 2021-11-05 20:57:19
@desc: 
"""
import torch
import math
import torchvision as tv
import torch.nn.functional as F
import torch.utils.model_zoo
from jjzhk.config import DetectConfig
from lib.model.model_zoo import MODEL_ZOO
from lib.yolov1.utils import decoder
from model.base import ModelBase
from jjzhk.device import device
from lib.backbone.resnet import ResNet, Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


@MODEL_ZOO.register()
def yolov1_resnet(cfg: DetectConfig):
    return MyResNet(cfg)


@MODEL_ZOO.register()
def yolov1_resnet50(cfg: DetectConfig):
    return ResNet50(cfg)


@MODEL_ZOO.register()
def yolov1_vgg16(cfg: DetectConfig):
    return VGG16(cfg)


class ResNet50(ModelBase):
    def __init__(self, cfg):
        super(ResNet50, self).__init__(cfg)
        self.base_model = tv.models.resnet50(pretrained=False)

        self.conv1 = torch.nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.conv3 = torch.nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.relu3 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = torch.nn.Conv2d(256, 2 * 5 + cfg['dataset']['classno'], kernel_size=(1, 1), stride=(1, 1))

    def load_backbone_weights(self, weights):
        self.base_model.load_state_dict(weights)

    def forward(self, x, **kwargs):
        output = x
        output = self.base_model.conv1(output)
        output = self.base_model.bn1(output)
        output = self.base_model.relu(output)
        output = self.base_model.maxpool(output)
        output = self.base_model.layer1(output)
        output = self.base_model.layer2(output)
        output = self.base_model.layer3(output)
        output = self.base_model.layer4(output)

        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        if self.cfg['net']['cell_number'] == 7:
            output = self.maxpool(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = torch.sigmoid(output)
        output = output.permute(0, 2, 3, 1)

        return output

    def get_eval_predictions(self, sampler, **kwargs):
        images, info = sampler[0], sampler[2]
        images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
        detections = self.model(images)

        result = []
        for detection in detections:
            w, h = info['width'], info['height']
            boxes, cls_indexs, probs = decoder(detection)

            for i, box in enumerate(boxes):
                # if cls_indexs[i] == 19:
                #     print("OK")

                prob = probs[i]
                prob = float(prob)
                if prob >= self.cfg['base']['conf_threshold']:
                    x1 = int(box[0] * w)
                    x2 = int(box[2] * w)
                    y1 = int(box[1] * h)
                    y2 = int(box[3] * h)

                    result.append([(x1, y1), (x2, y2), int(cls_indexs[i]), self.cfg.classname(int(cls_indexs[i])), prob])

        re_boxes = [[] for _ in range(len(self.cfg.class_keys()) + 1)]
        for (x1, y1), (x2, y2), class_id, class_name, prob in result: #image_id is actually image_path
            re_boxes[class_id+1].append([x1, y1, x2, y2, prob])

        return re_boxes

    def get_test_predict(self, image, info, **kwargs):
        result = []
        with torch.no_grad():
            img = torch.autograd.Variable(torch.FloatTensor(image))
            img = img.to(device)
            img_id = info[0]['img_id']
            pred = self.forward(img)  # 1x7x7x30
            pred = pred.cpu()
            boxes, cls_indexs, probs = decoder(pred)
            w = info[0]['width']
            h = info[0]['height']

            for i, box in enumerate(boxes):
                prob = probs[i]
                prob = float(prob)
                if (prob >= self.cfg['base']['conf_threshold']):
                    x1 = int(box[0] * w)
                    x2 = int(box[2] * w)
                    y1 = int(box[1] * h)
                    y2 = int(box[3] * h)
                    cls_index = cls_indexs[i]
                    cls_index = int(cls_index)  # convert LongTensor to int

                    result.append([
                        (x1, y1),
                        (x2, y2),
                        self.cfg.classname(cls_index),
                        img_id,
                        prob
                    ])

        return result

    def load_init_weights(self, weights):
        self.load_state_dict(weights["state_dict"])


class MyResNet(ModelBase):
    def __init__(self,cfg):
        super(MyResNet, self).__init__(cfg)
        self.inplanes = 64
        layers = [3, 4, 6, 3]
        num_classes = 1470
        self.cfg = cfg
        self.resnet = ResNet(Bottleneck, layers)

        self.layer5 = self._make_detnet_layer(in_channels=2048)

        self.conv_end = torch.nn.Conv2d(256, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                        bias=False)
        self.bn_end = torch.nn.BatchNorm2d(30)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_backbone_weights(self, weights):
        resnet = tv.models.resnet50(pretrained=False)
        resnet.load_state_dict(weights)
        new_state_dict = resnet.state_dict()
        dd = self.resnet.resnet.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        self.resnet.load_state_dict(dd)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)[2]
        x = self.layer5(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        if self.cfg['net']['cell_number'] == 7:
            x = torch.nn.MaxPool2d(kernel_size=(1,1), stride=(2,2))(x)
        x = torch.sigmoid(x)  # 归一化到0-1
        # x = x.view(-1,7,7,30)
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,30)

        return x

    def load_init_weights(self, weights):
        self.load_state_dict(weights['state_dict'])

    def get_eval_predictions(self, sampler, **kwargs):
        images, info = sampler[0], sampler[2]
        images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
        detections = self.model(images)

        result = []
        for detection in detections:
            w, h = info['width'], info['height']
            boxes, cls_indexs, probs = decoder(detection)

            for i, box in enumerate(boxes):
                # if cls_indexs[i] == 19:
                #     print("OK")

                prob = probs[i]
                prob = float(prob)
                if prob >= self.cfg['base']['conf_threshold']:
                    x1 = int(box[0] * w)
                    x2 = int(box[2] * w)
                    y1 = int(box[1] * h)
                    y2 = int(box[3] * h)

                    result.append(
                        [(x1, y1), (x2, y2), int(cls_indexs[i]), self.cfg.classname(int(cls_indexs[i])), prob])

        re_boxes = [[] for _ in range(len(self.cfg.class_keys()) + 1)]
        for (x1, y1), (x2, y2), class_id, class_name, prob in result:  # image_id is actually image_path
            re_boxes[class_id + 1].append([x1, y1, x2, y2, prob])

        return re_boxes

    def get_test_predict(self, image, info, **kwargs):
        result = []
        with torch.no_grad():
            img = torch.autograd.Variable(torch.FloatTensor(image))
            img = img.to(device)
            img_id = info[0]['img_id']
            pred = self.forward(img)  # 1x7x7x30
            pred = pred.cpu()
            boxes, cls_indexs, probs = decoder(pred)
            w = info[0]['width']
            h = info[0]['height']

            for i, box in enumerate(boxes):
                prob = probs[i]
                prob = float(prob)
                if (prob >= self.cfg['base']['conf_threshold']):
                    x1 = int(box[0] * w)
                    x2 = int(box[2] * w)
                    y1 = int(box[1] * h)
                    y2 = int(box[3] * h)
                    cls_index = cls_indexs[i]
                    cls_index = int(cls_index)  # convert LongTensor to int

                    result.append([
                        (x1, y1),
                        (x2, y2),
                        self.cfg.classname(cls_index),
                        img_id,
                        prob
                    ])

        return result


class VGG16(torch.nn.Module):
    def __init__(self, cfg: DetectConfig):
        super(VGG16, self).__init__()
        self.cfg = cfg
        self.base_model = tv.models.vgg16_bn(pretrained=True)

        self.conv1 = torch.nn.Conv2d(512, 30, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(30)

    def forward(self, input):
        output = input
        output = self.base_model.features(output) # bs, 512, 14, 14

        if self.cfg['net']['cell_number'] == 7:
            output = self.base_model.avgpool(output)

        output = self.conv1(output)
        output = self.bn1(output)

        output = torch.sigmoid(output)
        output = output.permute(0, 2, 3, 1)

        return output


class detnet_bottleneck(torch.nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride),
                                     padding=(2, 2), bias=False, dilation=(2, 2))
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion * planes, kernel_size=(1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion * planes)

        self.downsample = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1),
                                stride=(stride, stride), bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out




