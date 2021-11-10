"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov1_solver.py
@time: 2021-11-10 11:13:20
@desc: 
"""
import numpy as np
import torch
import torch.utils.data
import os
from jjzhk.device import device
import lib.model as m
import lib.yolov1 as y
from lib.utils.solver import Solver
from jjzhk.drawseg import BaseDrawSeg
from jjzhk.progressbar import ProgressBar
import lib.loss as l


class Yolov1Solver(Solver):
    def __init__(self, cfg, model_name):
        super(Yolov1Solver, self).__init__(cfg, model_name)

    #region virtual
    def get_model(self, model_name):
        return m.get_model("yolov1_" + model_name, self.cfg)

    def init_others(self):
        pass

    def load_check_point(self, weights, justInitBase=False):
        if justInitBase:
            pass
        else:
            weights_json = torch.load(weights)
            print(weights_json.keys())
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
        draw = BaseDrawSeg(cfg=self.cfg,
                           output=os.path.join(self._test_path_, str(epoch)))
        bar = ProgressBar(1, len(self._test_loader_), "Detection")
        for index, (image, _, info) in enumerate(self._test_loader_):
            with torch.no_grad():
                img = torch.autograd.Variable(torch.FloatTensor(image))
                img = img.to(device)
                img_id = info[0]['img_id']
                pred = self.model(img)  # 1x7x7x30
                pred = pred.cpu()
                boxes, cls_indexs, probs = y.decoder(pred)
                w = info[0]['width']
                h = info[0]['height']
                result = []
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
                            self.cfg.classname(cls_index + 1),
                            img_id,
                            prob
                        ])

            image = draw.draw_image(param={
                "Image": os.path.join(self.cfg['dataset']['test_root'], "Images", "%s.jpg" % img_id),
                "Boxes": result,
                "ImageName": img_id
            }, draw_type=0)
            bar.show(1)

    def eval_epoch(self, epoch, model):
        eval_model = y.YOLOV1Eval(self.cfg, model)

        mAP, info = eval_model.calculateMAP(self._eval_loader_,
                                            os.path.join(self._eval_path_, str(epoch)))
        self.logger.save_eval(epoch, mAP)

        headers = ['class name', 'AP']
        table = []
        for i, cls_name in enumerate(self.cfg.keys()):
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
        return l.get_loss('yololoss', self.cfg)

    def before_train(self):
        pass

    def change_lr(self, max_epochs, current_epoch, lr) -> float:
        lt = [int(ele.replace('scheduler_', '')) for ele in self.cfg['train'].keys() if ele.startswith('scheduler_')]
        lt = sorted(lt)

        learning_rate = lr

        if current_epoch in lt:
            learning_rate = self.cfg['train']['scheduler_%d' % current_epoch]

        # if current_epoch == 30:
        #     learning_rate = 0.0001
        # if current_epoch == 40:
        #     learning_rate = 0.00001

        for param_group in self._optimizer_.param_groups:
            param_group['lr'] = lr

        return learning_rate

    def train_epoch(self, epoch, bar, newir) -> tuple:
        avg_loss = 0
        time = 0
        i = 0

        for index, (images, target, info) in enumerate(self._train_loader_):
            images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
            target = torch.autograd.Variable(torch.FloatTensor(target)).to(device)
            # images, target = images.to(device), target.to(device)

            pred = self.model(images)
            loss = self._criterion_(pred, target)
            avg_loss += loss.item()

            self._optimizer_.zero_grad()
            loss.backward()
            self._optimizer_.step()

            time = bar.show(epoch, loss.item(), avg_loss / (index + 1), newir)

        return avg_loss / (i + 1), time
    #endregion

    #region private

    #endregion