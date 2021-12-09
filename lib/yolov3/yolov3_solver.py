"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov1_solver.py
@time: 2021-11-10 11:13:20
@desc: 
"""
from lib.utils.eval import EvalObj
import torch
import torch.utils.data
import os
from jjzhk.device import device
import lib.model as m
from lib.utils.solver import Solver
from jjzhk.drawseg import BaseDrawSeg
from jjzhk.progressbar import ProgressBar
import lib.yolov2 as y



class Yolov3Solver(Solver):
    def __init__(self, cfg, model_name):
        super(Yolov3Solver, self).__init__(cfg, model_name)

    #region virtual
    def get_model(self, model_name):
        return m.get_model("yolov3_" + model_name, self.cfg)

    def init_others(self):
        pass

    def load_check_point(self, weights, justInitBase=False):
        if justInitBase:
            pass
        else:
            weights_json = torch.load(weights, map_location=device)
            self.model.load_state_dict(weights_json)

    def init_test_loader(self):
        return torch.utils.data.DataLoader(self._test_dataset_,
                                           batch_size=self.cfg['test']['batch_size'],
                                           collate_fn=self._test_dataset_.collater)

    def init_eval_loader(self) -> object:
        return torch.utils.data.DataLoader(self._eval_dataset_,
                                           batch_size=self.cfg['eval']['batch_size'],
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=self._eval_dataset_.collater)

    def init_train_loader(self) -> object:
        return torch.utils.data.DataLoader(self._train_dataset_,
                                           batch_size=self.cfg['train']['batch_size'],
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=self._train_dataset_.collater)

    def test_epoch(self, epoch, model):
        self.model.set_grid(self.cfg['test']['imagesize'][0])
        draw = BaseDrawSeg(cfg=self.cfg,
                           output=os.path.join(self._test_path_, str(epoch)))
        bar = ProgressBar(1, len(self._test_loader_), "Detection")
        for index, (image, _, info) in enumerate(self._test_loader_):
            boxes = self.model.get_test_predict(image, info)
            img_id = info[0]["img_id"]
            image = draw.draw_image(param={
                "Image": os.path.join(self.cfg['dataset']['test_root'], "Images", "%s.jpg" % img_id),
                "Boxes": boxes,
                "ImageName": img_id
            }, draw_type=0)
            bar.show(1)

    def eval_epoch(self, epoch, model):
        self.model.set_grid(self.cfg['eval']['imagesize'][0])
        eval_model = EvalObj(self.cfg, model)

        mAP, info = eval_model.calculateMAP(self._eval_loader_,
                                            os.path.join(self._eval_path_, str(epoch)))
        self.logger.save_eval(epoch, mAP)

        headers = ['class name', 'AP']
        table = []
        for i, cls_name in enumerate(self.cfg.class_keys()):
            table.append([cls_name, info[cls_name]])

        self.logger.save_eval_txt_file(epoch, table, headers)

    def get_train_parameters(self) -> list:
        params = []
        params_dict = dict(self.model.named_parameters())
        for key, value in params_dict.items():
            if key.startswith('features'):
                params += [{'params': [value], 'lr': self.cfg['train']['learning_rate'] * 1}]
            else:
                params += [{'params': [value], 'lr': self.cfg['train']['learning_rate']}]

        return params

    def get_loss(self):
        pass

    def before_train(self):
        pass

    def change_lr(self, max_epochs, current_epoch, lr) -> float:
        base_lr = self.cfg['train']['learning_rate']
        tmp_lr = base_lr
        lr_epoch = self.cfg['train']['lr_epoch']
        lr_epoch = [int(x) for x in lr_epoch.split(',')]

        if current_epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            for param_group in self._optimizer_.param_groups:
                param_group['lr'] = tmp_lr

        return tmp_lr

    def change_lr_iter(self, current_epoch, iter_index):
        wp_epoch = self.cfg['train']['wp_epoch']
        base_lr = self.cfg['train']['learning_rate']
        epoch_size = len(self._train_dataset_) // (self.cfg['train']['batch_size'] * 1)
        if current_epoch < wp_epoch:
            tmp_lr = base_lr * pow((iter_index+ current_epoch * epoch_size)*1. / (wp_epoch * epoch_size), 4)
            for param_group in self._optimizer_.param_groups:
                param_group['lr'] = tmp_lr
        elif current_epoch == wp_epoch and iter_index == 0:
            tmp_lr = base_lr
            for param_group in self._optimizer_.param_groups:
                param_group['lr'] = tmp_lr

    def train_epoch(self, epoch, bar, newir) -> tuple:
        avg_loss = 0
        time = 0
        index = 0
        self.model.set_grid(self.cfg['train']['imagesize'][0])
        for index, (images, target, info) in enumerate(self._train_loader_):
            self.change_lr_iter(epoch, index)
            targets = [label.tolist() for label in target]

            targets = y.multi_gt_creator(input_size=self.cfg['train']['imagesize'][0],
                                         strides=self.model.stride,
                                         label_lists=targets,
                                         anchor_size=self.cfg['base']['anchors'])

            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            conf_loss, cls_loss, box_loss, iou_loss = self.model(images, target=targets, trainable=True)

            # compute loss
            total_loss = conf_loss + cls_loss + box_loss + iou_loss

            avg_loss += total_loss.item()

            # check NAN for loss
            if torch.isnan(total_loss):
                continue

            # backprop
            total_loss.backward()
            self._optimizer_.step()
            self._optimizer_.zero_grad()

            time = bar.show(epoch, total_loss.item(), avg_loss / (index + 1), newir)

        return avg_loss / (index + 1), time
    #endregion

    #region private

    #endregion