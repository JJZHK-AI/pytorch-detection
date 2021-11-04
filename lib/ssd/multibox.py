'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: multibox.py
@time: 2019-07-22 11:50:48
@desc: 
'''
import torch
from lib.loss.loss_zoo import LOSS_ZOO
from lib.ssd.utils import match, match_ious, log_sum_exp, decode, bbox_overlaps_iou
from lib.ssd.utils import bbox_overlaps_giou, bbox_overlaps_ciou, bbox_overlaps_diou


@LOSS_ZOO.register()
def multi(cfg, priors):
    return MultiBoxLoss(cfg, priors)


class MultiBoxLoss(torch.nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors):
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.use_gpu = torch.cuda.is_available()
        self.num_classes = cfg['dataset']['classno']
        # self.background_label = cfg.BASE.BACKGROUND_LABEL
        self.negpos_ratio = float(cfg['base']['negpos_ratio'])
        self.threshold = float(cfg['base']['conf_threshold'])
        self.variance = cfg['base']['variance']
        self.priors = priors
        self.loss = cfg['base']['loss']
        self.ious = IouLoss(pred_mode='Center', size_sum=True, variances=self.variance,
                             losstype=self.loss)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0) #batch size的个数
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        '''
        匹配不同feature map生成的锚结果与真值框,将结果保存
        '''
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            if self.loss == 'SmoothL1':
                match     (self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            else:
                match_ious(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets loc_t, conf_t保存一个batch中的坐标偏移值  和 分类误差
        loc_t = torch.autograd.Variable(loc_t, requires_grad=False)
        conf_t = torch.autograd.Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        # num_pos = pos.sum()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)

        if self.loss == 'SmoothL1':
            loss_l = torch.nn.functional.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        else:
            giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
            loss_l = self.ious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # Hard Negative Mining
        '''
        1. 根据正样本的个数和正负比例，确定负样本的个数，negative_keep
        2. 找到confidence loss最大的negative_keep个负样本，计算他们的分类损失之和
        3. 计算正样本的分类损失之和，分类损失是正样本和负样本的损失和
        4. 计算正样本的位置损失localization loss.无法计算负样本位置损失
        5. 对回归损失和位置损失求和
        '''
        loss_c = loss_c.view(pos.size()[0], pos.size()[1]) #add line
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = torch.nn.functional.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l/=N
        loss_c/=N
        return loss_l + loss_c


class IouLoss(torch.nn.Module):
    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        decoded_boxes = loc_p # 我觉得这里不应该解码，因为detector里面会进行解码的。

        # if self.pred_mode == 'Center':
        #     decoded_boxes = decode(loc_p, prior_data, self.variances)
        # else:
        #     decoded_boxes = loc_p

        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        elif self.loss == 'Giou':
            loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
        elif self.loss == 'Diou':
            loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
        else: # 'Ciou'
            loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss


