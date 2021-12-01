"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: ssd_solver.py
@time: 2021-10-27 11:38:29
@desc: 
"""
import torch
import torch.utils.data
import os
from lib.utils.solver import Solver
import torch.utils.model_zoo
import lib.ssd as s
import lib.model as m
import lib.loss as l
import itertools
import math
from jjzhk.device import device
from lib.utils.eval import EvalObj
from jjzhk.drawseg import BaseDrawSeg
from jjzhk.progressbar import ProgressBar


class SSDSolver(Solver):
    def __init__(self, cfg, model_name):
        super(SSDSolver, self).__init__(cfg, model_name)

    #region virtual to override
    def get_model(self, model_name):
        return m.get_model(model_name, self.cfg)

    def init_others(self):
        self.priors = None
        self.detector = None

        feature_maps = self._forward_features_size(self.model, self.cfg['net']['imagesize'])

        self.priorbox = PriorBox(image_size=self.cfg['net']['imagesize'], feature_maps=feature_maps,
                                 aspect_ratios=self.cfg['net']['aspect_ratio'],
                                 scale=self.cfg['net']['sizes'], archor_stride=self.cfg['net']['steps'],
                                 clip=self.cfg['base']['clip'])

        self.priors = torch.autograd.Variable(self.priorbox.forward())
        self.detector = Detect(self.cfg, self.priors)

    def init_test_loader(self):
        test_sampler = self._make_data_sampler_(self._test_dataset_, False)
        test_batch_sampler = self._make_batch_data_sampler_(test_sampler,
                                                            images_per_batch=self.cfg['test']['batch_size'],
                                                            drop_last=False)

        return torch.utils.data.DataLoader(dataset=self._test_dataset_,
                                                         batch_sampler=test_batch_sampler,
                                                         num_workers=0,
                                                         pin_memory=True,
                                                         collate_fn=self._init_detection_collate_)

    def init_eval_loader(self):
        eval_sampler = self._make_data_sampler_(self._eval_dataset_, False)
        eval_batch_sampler = self._make_batch_data_sampler_(eval_sampler,
                                                            images_per_batch=self.cfg['eval']['batch_size'],
                                                            drop_last=False)
        return torch.utils.data.DataLoader(dataset=self._eval_dataset_,
                                                         batch_sampler=eval_batch_sampler,
                                                         num_workers=0,
                                                         pin_memory=True,
                                                         collate_fn=self._init_detection_collate_)

    def init_train_loader(self):
        train_sampler = self._make_data_sampler_(self._train_dataset_, False)
        train_batch_sampler = self._make_batch_data_sampler_(train_sampler,
                                                           images_per_batch=self.cfg['train']['batch_size'],
                                                           drop_last=False)
        return torch.utils.data.DataLoader(dataset=self._train_dataset_,
                                                          batch_sampler=train_batch_sampler,
                                                          num_workers=0,
                                                          pin_memory=True,
                                                          collate_fn=self._init_detection_collate_)

    def test_epoch(self, epoch, model):
        draw = BaseDrawSeg(cfg=self.cfg,
                           output=os.path.join(self._test_path_, str(epoch)))

        self.barCfg['total'] = len(self._test_loader_)
        bar = ProgressBar(1, len(self._test_loader_), "Detection")

        for i, sampler in enumerate(self._test_loader_):
            images, info = sampler['img'], sampler['info']
            boxes = self.model.get_test_predict(images, info, eval=False, detector=self.detector)
            for j, box in enumerate(boxes):
                image_id = info[j]["img_id"]
                filename = "%s.jpg" % image_id
                image = draw.draw_image(param={
                    "Image": os.path.join(self.cfg['dataset']['test_root'], "Images", filename),
                    "Boxes": box,
                    "ImageName": image_id
                }, draw_type=0)
            bar.show(1)

        return image

    def eval_epoch(self, epoch, model):
        eval_model = EvalObj(self.cfg, model)

        mAP, info = eval_model.calculateMAP(self._eval_loader_,
                                            os.path.join(self._eval_path_, str(epoch)),
                                            detector=self.detector)
        self.logger.save_eval(epoch, mAP)

        headers = ['class name', 'AP']
        table = []
        for i, cls_name in enumerate(self.cfg.class_keys()):
            table.append([cls_name, info[cls_name]])

        self.logger.save_eval_txt_file(epoch, table, headers)

    def get_train_parameters(self):
        trainable_scope = self.cfg['train']['scope']
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param if len(trainable_param) else self.model.parameters()

    def get_loss(self):
        return l.get_loss('multi', self.cfg, priors=self.priors if hasattr(self, 'priors') else None)

    def change_lr(self, max_epochs, current_epoch, lr):
        newir = lr
        if current_epoch > self.cfg['train']['warm_up_epochs'] and \
                self._lr_scheduler_ is not None:
            self._lr_scheduler_.step(current_epoch - self.cfg['train']['warm_up_epochs'])
            for param_group in self._optimizer_.param_groups:
                newir = param_group['lr']
        else:
            if hasattr(self.cfg['train'], 'STEP'):
                steps = self.cfg['train']['step']
                attributes = sorted([int(attr) for attr in steps.__dict__ if attr != '_content'], reverse=True)
                flag = True
                for s in attributes:
                    if current_epoch >= s and flag:
                        for param_group in self._optimizer_.param_groups:
                            param_group['lr'] = float(steps.__dict__[str(s)])
                        newir = float(steps.__dict__[str(s)])
                        flag = False
        return newir

    def train_epoch(self, epoch, bar, newir):
        avg_loss = 0
        time = 0
        i = 0

        for i, inputs in enumerate(self._train_loader_):
            # {'img': np.stack(imgs, 0), 'annot': targets, 'info': infos}
            images, target, _ = inputs['img'], inputs['annot'], inputs['info']
            images = torch.autograd.Variable(torch.FloatTensor(images)).to(device)
            targets = [torch.autograd.Variable(torch.FloatTensor(anno)).to(device) for anno in target]
            pred = self.model(images, phase='train', target=target)

            if self._criterion_ is None:
                loss = pred
            else:
                loss = self._criterion_(pred, targets)

            avg_loss += loss.item()

            self._optimizer_.zero_grad()
            loss.backward()
            self._optimizer_.step()

            time = bar.show(epoch, loss.item(), avg_loss / (i + 1), newir)

        return avg_loss / (i + 1), time
    #endregion

    #region private
    def _forward_features_size(self, model, img_size):
        model.eval()
        x = torch.rand(1, 3, img_size[0], img_size[1])
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            x = torch.autograd.Variable(x)
        feature_maps = model(x, phase='feature')
        return [(o.size()[2], o.size()[3]) for o in feature_maps]

    def _make_data_sampler_(self, dataset, shuffle):
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler

    def _make_batch_data_sampler_(self, sampler, images_per_batch, num_iters=None, start_iter=0, drop_last=True):
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_batch, drop_last=drop_last)
        if num_iters is not None:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        return batch_sampler

    def _init_detection_collate_(self, batch):
        if self.phase == 'train':
            return self._train_dataset_.collater(batch)
        elif self.phase == 'eval':
            return self._eval_dataset_.collater(batch)
        else:
            return self._test_dataset_.collater(batch)
    #endregion


class Detect(torch.autograd.Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg, priors):
        self.num_classes = cfg['dataset']['classno']
        # self.background_label = cfg.BASE.BACKGROUND_LABEL
        self.conf_thresh = float(cfg['base']['conf_threshold'])
        self.nms_thresh = float(cfg['base']['iou_threshold'])
        self.top_k = cfg['detect']['max_detections']
        self.variance = cfg['base']['variance']
        self.cfg = cfg
        self.priors = priors

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = self.priors.data

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        # self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.transpose(0, 1).squeeze().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            # self.output.expand_(num, self.num_classes, self.top_k, 5)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = s.decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh).nonzero(as_tuple=False).view(-1)
                if c_mask.dim() == 0:
                    continue
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                boxes = decoded_boxes[c_mask, :]
                if self.cfg['base']['nms'] == 'diounms':
                    ids, count = s.diounms(boxes, scores, self.nms_thresh, self.top_k)
                elif self.cfg['base']['nms'] == 'nms':
                    ids, count = s.nms(boxes, scores, self.nms_thresh, self.top_k)
                elif self.cfg['base']['nms'] == 'soft_nms':
                    pass

                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    '''
    一般来说，先验框Prior Box有8732个，38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4，但是本例中一般为11620个
    38*38*6+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4
    '''

    def __init__(self, image_size, feature_maps, aspect_ratios, scale, archor_stride=None, archor_offest=None,
                 clip=True):
        super(PriorBox, self).__init__()
        self.image_size = image_size  # 图片size
        self.feature_maps = feature_maps  # 特征图各层的size
        # aspect_ratios : [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
        self.aspect_ratios = aspect_ratios  # 先验框的比率
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(aspect_ratios)  #
        self.clip = clip
        # scale value 每张特征图的最大尺寸和最小尺寸

        # scale : [[30, 30], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]]
        if isinstance(scale[0], list):  # scale = size
            self.scales = [min(s[0] / self.image_size[0], s[1] / self.image_size[1]) for s in scale]
        elif isinstance(scale[0], float) and len(scale) == 2:
            num_layers = len(feature_maps)
            min_scale, max_scale = scale
            self.scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [
                1.0]

        # 感受野大小，即相对于原图的缩小倍数
        # archor_stride = [[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]]
        self.steps = [(self.image_size[0] / steps[0], self.image_size[0] / steps[1]) for steps in archor_stride]

    # def forward(self):
    #     mean = []
    #     for k, f in enumerate(self.feature_maps):  # 存放的是feature map的尺寸：38 19 10 5 3 1
    #         for i, j in itertools.product(range(f[0]), range(f[1])):  # 对于一个feature map上的所有的位置
    #             f_k = self.steps[k]  # steps=[8,16,32,64,100,300] 得到feature map的尺寸
    #             # unit center x,y
    #             cx = (j + 0.5) / f_k[1]
    #             cy = (i + 0.5) / f_k[0]
    #             # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.
    #
    #             # aspect_ratio: 1
    #             # rel size: min_size
    #             s_k = self.scales[k]
    #             mean += [cx, cy, s_k, s_k]
    #
    #             # aspect_ratio: 1
    #             # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
    #             # rel size: sqrt(s_k * s_(k+1))
    #             s_k_prime = math.sqrt(s_k * self.scales[k+1])
    #             mean += [cx, cy, s_k_prime, s_k_prime]
    #
    #             # rest of aspect ratios
    #             for ar in self.aspect_ratios[k]:
    #                 mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
    #                 mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
    #     output = torch.Tensor(mean).view(-1, 4)
    #     if self.clip:
    #         output.clamp_(max=1, min=0)
    #     return output

    def forward(self):
        mean = []
        # l = 0
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f[0]), range(f[1])):
                '''
                cx, cy为当前feature map的当前像素的中心坐标
                '''
                cx = (j + 0.5) / self.steps[k][1]
                cy = (i + 0.5) / self.steps[k][0]
                s_k = self.scales[k]

                for ar in self.aspect_ratios[k]:
                    if isinstance(ar, int):
                        if ar == 1:
                            # aspect_ratio: 1 Min size
                            mean += [cx, cy, s_k, s_k]

                            # aspect_ratio: 1 Max size
                            s_k_prime = math.sqrt(s_k * self.scales[k + 1])
                            mean += [cx, cy, s_k_prime, s_k_prime]
                        else:
                            ar_sqrt = math.sqrt(ar)
                            mean += [cx, cy, s_k * ar_sqrt, s_k / ar_sqrt]
                            mean += [cx, cy, s_k / ar_sqrt, s_k * ar_sqrt]
                    elif isinstance(ar, list):
                        mean += [cx, cy, s_k * ar[0], s_k * ar[1]]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
