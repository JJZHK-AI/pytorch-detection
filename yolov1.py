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
import torch.utils.data
from lib.yolov1.model import resnet50
import torchvision
import numpy as np
import os
from jjzhk.config import DetectConfig
from lib.yolov1.dataset import VOCDetection
from jjzhk.device import device
from lib.yolov1.loss import yoloLoss
from jjzhk.progressbar import ProgressBar
from jjzhk.logger import Logger

ROOT = "/" #"/Users/jjzhk/data/"
learning_rate = 0.001
num_epochs = 50
batch_size = 24
net = resnet50()
resnet = torchvision.models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in dd.keys() and not k.startswith('fc'):
        print('yes')
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)
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

config = DetectConfig("cfg")
config.loadfile("voc.cfg")
config['dataset']['root'] = os.path.join(ROOT, config['dataset']['root'])
train_dataset = VOCDetection(config, phase="train")
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
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

        bar.show(index, loss.item(), total_loss / (index + 1))

    logger.info("epoch {epoch}: lr-{lr}", epoch=epoch, lr=learning_rate)
    torch.save(net.state_dict(), "%d.pth" % epoch)