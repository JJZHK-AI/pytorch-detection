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
import cv2
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
        self.model.load_state_dict(torch.load(os.path.join(torch.hub.get_dir(), 'checkpoints', weights), map_location='cpu'))

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
        for index, (image, info) in enumerate(self._test_loader_):
            img = image.to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=False)[0]
            pred = y.non_max_suppression(pred, self.cfg['base']['conf_threshold'],
                                       self.cfg['base']['iou_threshold'],
                                       classes=None, agnostic=False)
            imgid = info["img_id"][0]
            path = info["path"][0]
            im0 = cv2.imread(path)
            s = ''

            for i, det in enumerate(pred):
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    det[:, :4] = y.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.cfg.classname(int(c)))  # add to string

                    # Write results
                    result_box = []
                    for *xyxy, conf, cls in det:
                        result_box.append(
                            [
                                (xyxy[0].item(), xyxy[1].item()),
                                (xyxy[2].item(), xyxy[3].item()),
                                self.cfg.classname(int(cls)),
                                "", conf.item()
                            ]
                        )
                    draw = BaseDrawSeg(cfg=self.cfg,
                                       output=os.path.join(self._test_path_, str(epoch)))
                    (_, filename) = os.path.split(path)
                    draw.draw_image(param={
                        "Image": path,
                        "Boxes": result_box,
                        "ImageName": filename.replace(".jpg", "")
                    })

            print('%s Done' % s)

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
        pass

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