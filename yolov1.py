"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: train.py
@time: 2021-11-05 20:56:41
@desc: 
"""
import os

import numpy as np
import torch.utils.data
import torchvision
from jjzhk.config import DetectConfig
from jjzhk.device import device
from jjzhk.drawseg import BaseDrawSeg
from jjzhk.logger import Logger
from jjzhk.progressbar import ProgressBar

from lib.yolov1.dataset import VOCDetection
from lib.yolov1.loss import yoloLoss
from lib.yolov1.model import resnet50, ResNet50
from lib.yolov1.utils import decoder

phase = 'train'
ROOT = "/Users/jjzhk/data/"
learning_rate = 0.0005 # 0.001
num_epochs = 50
batch_size = 32 # 24
net = ResNet50() #resnet50() #
config = DetectConfig("cfg")
config.loadfile("voc.cfg")
config['dataset']['root'] = os.path.join(ROOT, config['dataset']['root'])

if phase == 'train':
    # resnet = torchvision.models.resnet50(pretrained=True)
    # new_state_dict = resnet.state_dict()
    # dd = net.state_dict()
    # for k in new_state_dict.keys():
    #     if k in dd.keys() and not k.startswith('fc'):
    #         dd[k] = new_state_dict[k]
    # net.load_state_dict(dd)
    net.to(device)

    net.train()
    # different learning rate
    params=[]
    params_dict = dict(net.named_parameters())
    for key,value in params_dict.items():
        if key.startswith('features'):
            params += [{'params':[value],'lr':learning_rate*1}]
        else:
            params += [{'params':[value],'lr':learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train_dataset = VOCDetection(config, phase="train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    # test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
    # test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
    # test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')

    num_iter = 0
    best_test_loss = np.inf
    criterion = yoloLoss(7,2,5,0.5)
    bar = ProgressBar(num_epochs, len(train_loader), "Loss: %.4f, average_loss: %.4f")
    logger = Logger(output="logs", logger_name="yolov1", handlers='f',formatter="")
    for epoch in range(1, num_epochs + 1):
        net.train()
        if epoch == 30:
            learning_rate=0.0001
        if epoch == 40:
            learning_rate=0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        total_loss = 0.
        for index, (images, target) in enumerate(train_loader):
            images = torch.autograd.Variable(images)
            target = torch.autograd.Variable(target)
            images, target = images.to(device), target.to(device)

            pred = net(images)
            loss = criterion(pred, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.show(epoch, loss.item(), total_loss / (index + 1))

        logger.info("epoch {epoch}: lr-{lr}", epoch=epoch, lr=learning_rate)
        torch.save(net.state_dict(), "%d.pth" % epoch)
elif phase == 'eval':
    pass
elif phase == 'test':
    net.load_state_dict(torch.load("weights/44.pth", map_location='cpu'))
    net.to(device)
    net.eval()
    test_dataset = VOCDetection(config, phase="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    draw = BaseDrawSeg(cfg=config, output="output/")
    bar = ProgressBar(num_epochs, len(test_loader), "Detection")
    for index, (image, info) in enumerate(test_loader):
        img = torch.autograd.Variable(image, volatile=True)
        img = img.to(device)
        img_id = info['img_id'][0]
        pred = net(img)  # 1x7x7x30
        pred = pred.cpu()
        boxes, cls_indexs, probs = decoder(pred)
        w = info['width'].item()
        h = info['height'].item()
        result = []
        for i, box in enumerate(boxes):
            prob = probs[i]
            prob = float(prob)
            if (prob >= 0.3):
                x1 = int(box[0] * w)
                x2 = int(box[2] * w)
                y1 = int(box[1] * h)
                y2 = int(box[3] * h)
                cls_index = cls_indexs[i]
                cls_index = int(cls_index)  # convert LongTensor to int

                result.append([
                    (x1, y1),
                    (x2, y2),
                    config.classname(cls_index+1),
                    img_id,
                    prob
                ])

        image = draw.draw_image(param={
            "Image": os.path.join(config['dataset']['test_root'], "Images", "%s.jpg" % img_id),
            "Boxes": result,
            "ImageName": img_id
        }, draw_type=0)
        bar.show(1)






