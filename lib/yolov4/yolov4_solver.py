"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov4_solver.py
@time: 2021-10-27 13:20:20
@desc: 
"""
import torch
import torch.utils.data
import os
from jjzhk.device import device
from lib.utils.solver import Solver
import torch.utils.model_zoo
import lib.yolov4 as y
import glob
from pathlib import Path
from jjzhk.drawseg import BaseDrawSeg
from lib.yolov4.darknet import Darknet
from jjzhk.progressbar import ProgressBar
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YOLOV4Solver(Solver):
    def __init__(self, cfg, model_name):
        super(YOLOV4Solver, self).__init__(cfg, model_name)

    #region virtual to override
    def get_model(self, model_name):
        return Darknet(self.cfg).to(device)

    def init_others(self):
        pass

    def load_check_point(self, weights, justInitBase=False):
        self.model.load_state_dict(torch.load(weights, map_location=device))

    def init_test_loader(self):
        return torch.utils.data.DataLoader(dataset=self._test_dataset_,
                                           batch_size=1, num_workers=0,
                                                         pin_memory=True)

    def init_eval_loader(self) -> object:
        batch_size = 32
        workers = 8
        batch_size = min(batch_size, len(self._eval_dataset_))
        nw = min([os.cpu_count() // 1, batch_size if batch_size > 1 else 0, workers])  # number of workers
        sampler = None
        return y.InfiniteDataLoader(self._eval_dataset_,batch_size=32,
                             num_workers=0, sampler=sampler, pin_memory=False,
                             collate_fn=YOLOV4Solver.collate_fn)

    def init_train_loader(self) -> object:
        pass

    def test_epoch(self, epoch, model):
        draw = BaseDrawSeg(cfg=self.cfg, output=os.path.join(self._test_path_, str(epoch)))
        bar = ProgressBar(1, len(self._test_loader_), "Detection")
        for index, (image, info) in enumerate(self._test_loader_):
            boxes = self.model.get_test_predict(image, info)
            img_id = info["img_id"][0]
            image = draw.draw_image(param={
                "Image": os.path.join(self.cfg['dataset']['test_root'], "Images", "%s.jpg" % img_id),
                "Boxes": boxes,
                "ImageName": img_id
            }, draw_type=0)
            bar.show(1)

    def eval_epoch(self, epoch, model):
        seen = 0
        jdict = []
        bar = ProgressBar(1, len(self._eval_loader_), "")

        for index, (image, targets, info, shapes) in enumerate(self._eval_loader_):
            img = image.to(device, non_blocking=True)
            img = img.float()
            img /= 255
            targets = targets.to(device)
            nb, _, height, width = img.shape
            whwh = torch.Tensor([width, height, width, height]).to(device)

            with torch.no_grad():
                inf_out, train_out = self.model(img, augment=False)
                output = y.non_max_suppression(inf_out, conf_thres=self.cfg['base']['conf_threshold'],
                                               iou_thres=self.cfg['base']['iou_threshold'])

            for si, pred in enumerate(output):
                path = Path(info[si]['path'])
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1
                y.clip_coords(pred, (height, width))
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                y.scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = y.xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': self.cfg.converse_transfer_to_id(int(p[5])),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            bar.show(1)

        anno_json = glob.glob(os.path.join(self.cfg['dataset']['root'], "annotations", "instances_val2017.json"))[0]
        pred_json = os.path.join(self._eval_path_, "predictions.json")
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        anno = COCO(anno_json)
        pred = anno.loadRes(pred_json)
        eval = COCOeval(anno, pred, 'bbox')
        eval.params.imgIds = [int(Path(x).stem) for x in self._eval_dataset_.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]
        self.logger.save_eval(0, map)

    def get_train_parameters(self) -> list:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2.append(v)  # biases
            elif 'Conv2d.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'm.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'w.weight' in k:
                pg1.append(v)  # apply weight_decay
            else:
                pg0.append(v)  # all else

        return pg0

    def after_optimizer(self):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2.append(v)  # biases
            elif 'Conv2d.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'm.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'w.weight' in k:
                pg1.append(v)  # apply weight_decay
            else:
                pg0.append(v)  # all else

        self._optimizer_.add_param_group({'params': pg1, 'weight_decay': self.cfg['train']['weight_decay']})  # add pg1 with weight_decay
        self._optimizer_.add_param_group({'params': pg2})  # add pg2 (biases)

    def get_loss(self):
        pass

    def before_train(self):
        pass

    def change_lr(self, max_epochs, current_epoch, lr) -> float:
        pass

    def train_epoch(self, epoch, bar, newir) -> tuple:
        pass
    #endregion

    #region private
    @staticmethod
    def collate_fn(batch):
        img, label, info, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), info, shapes
    #endregion