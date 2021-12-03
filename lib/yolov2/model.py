"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: model.py
@time: 2021-11-12 14:23:58
@desc: 
"""
from lib.model.model_zoo import MODEL_ZOO
from jjzhk.config import DetectConfig
import torch
import lib.yolov2.tools as tools
import numpy as np
from jjzhk.device import device
from lib.backbone.backbone_layer import get_backbone
from model.base import ModelBase

@MODEL_ZOO.register()
def yolov2_darknet19(cfg: DetectConfig):
    return YOLOV2_Net(cfg, "darknet19")


class YOLOV2_Net(ModelBase):
    def __init__(self, cfg: DetectConfig, backbone_name):
        super(YOLOV2_Net, self).__init__(cfg)

        #region fields
        self.stride = 32
        self.device = device
        self.conf_thresh = self.cfg['base']['conf_threshold']
        self.nms_thresh = self.cfg['base']['nms_thresh']
        self.input_size = self.cfg['net']['imagesize'][0]
        self.backbone = get_backbone(cfg)
        self.num_anchors = len(self.cfg['base']['anchors'])
        self.anchor_size = torch.tensor(self.cfg['base']['anchors'])
        self.num_classes = self.cfg['dataset']['classno']
        self.grid_cell, self.all_anchor_wh = self.create_grid(self.input_size)

        self.convsets_1 = torch.nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)

        # prediction layer
        self.pred = torch.nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        #endregion

    def forward(self, x, **kwargs):
        target = kwargs.get('target')
        trainable = kwargs.get('trainable')

        # backbone主干网络
        _, c4, c5 = self.backbone(x)

        # head
        p5 = self.convsets_1(c5)

        # 处理c4特征
        p4 = self.reorg(self.route_layer(c4))

        # 融合
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

        # train
        if trainable:
            txtytwth_pred = txtytwth_pred.view(B, H * W, self.num_anchors, 4)
            # decode bbox
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # 计算预测框和真实框之间的IoU
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # 将IoU作为置信度的学习目标
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            txtytwth_pred = txtytwth_pred.view(B, H * W * self.num_anchors, 4)
            # 将IoU作为置信度的学习目标
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # 计算损失
            conf_loss, cls_loss, bbox_loss, iou_loss = tools.loss(pred_conf=conf_pred,
                                                                  pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,
                                                                  pred_iou=iou_pred,
                                                                  label=target
                                                                  )

            return conf_loss, cls_loss, bbox_loss, iou_loss

            # test
        else:
            txtytwth_pred = txtytwth_pred.view(B, H * W, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C],
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds

    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """将txtytwth预测换算成边界框的左上角点坐标和右下角点坐标 \n
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # 获得边界框的中心点坐标和宽高
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def get_eval_predictions(self, sampler, **kwargs):
        pass
        # images, info = sampler[0], sampler[2]
        # images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
        # detections = self.model(images)
        #
        # result = []
        # for detection in detections:
        #     w, h = info['width'], info['height']
        #     boxes, cls_indexs, probs = decoder(detection)
        #
        #     for i, box in enumerate(boxes):
        #         # if cls_indexs[i] == 19:
        #         #     print("OK")
        #
        #         prob = probs[i]
        #         prob = float(prob)
        #         if prob >= self.cfg['base']['conf_threshold']:
        #             x1 = int(box[0] * w)
        #             x2 = int(box[2] * w)
        #             y1 = int(box[1] * h)
        #             y2 = int(box[3] * h)
        #
        #             result.append([(x1, y1), (x2, y2), int(cls_indexs[i]), self.cfg.classname(int(cls_indexs[i])), prob])
        #
        # re_boxes = [[] for _ in range(len(self.cfg.class_keys()) + 1)]
        # for (x1, y1), (x2, y2), class_id, class_name, prob in result: #image_id is actually image_path
        #     re_boxes[class_id+1].append([x1, y1, x2, y2, prob])
        #
        # return re_boxes

    def get_test_predict(self, image, info, **kwargs):
        img_h, img_w = info[0]['height'], info[0]['width']
        scale = np.array([[img_w, img_h, img_w, img_h]])
        img = torch.from_numpy(image).to(device)
        bboxes, scores, cls_inds = self.forward(img, trainable=False)
        bboxes *= scale
        result = []
        img_id = info[0]['img_id']
        for i, box in enumerate(bboxes):
            prob = scores[i]
            prob = float(prob)
            if (prob >= self.cfg['base']['conf_threshold']):
                x1 = int(box[0])
                x2 = int(box[2])
                y1 = int(box[1])
                y2 = int(box[3])
                cls_index = cls_inds[i]
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
        self.load_state_dict(weights)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                torch.nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return self.convs(x)


class reorg_layer(torch.nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x

