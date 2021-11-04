'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: eval.py
@time: 2020-07-17 10:14:33
@desc: 
'''
import os
import numpy as np
from jjzhk.config import DetectConfig
from alive_progress import alive_bar


class EvalObj:
    def __init__(self, config: DetectConfig, model):
        self.cfg = config
        self.model = model
        self.barCfg = {'bar': 'halloween', 'spinner': None,
                       'receipt_text': False, 'monitor': True,
                       'stats': True, 'elapsed': False}

    def calculateMAP(self, loader, output_path, detector):
        all_boxes = [[[] for _ in range(len(loader))]
                     for _ in range(len(self.cfg['class_info'].keys()) + 1)]
        infos = []
        self.barCfg['total'] = len(loader)
        with alive_bar(title_length=20, title="Detection", **self.barCfg) as bar:
            for i, sampler in enumerate(loader):
                images, targets, info = sampler['img'], sampler['annot'], sampler['info']
                detections = self.model.get_detections(images, detector=detector)

                image_eval_boxes = self.model.get_eval_predictions(info, detections)

                for j, box in enumerate(image_eval_boxes):
                    all_boxes[j][i] = box

                infos.append(info[0])
                bar()
        print("calculating mAP...")
        return loader.dataset.evaluate_detections(all_boxes, output_path, infos)


def write_voc_results_file(cfg, output_dir, all_boxes, infos):
    for cls_ind, cls in enumerate(cfg.keys()):
        filename = get_voc_results_file_template(output_dir, cls)
        with open(filename, 'wt') as f:
            for im_ind, info in enumerate(infos):
                dets = np.array(all_boxes[cls_ind + 1][im_ind])
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(info['img_id'], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def get_voc_results_file_template(output_dir, cls):
    filename = 'det_%s.txt' % (cls)
    path = os.path.join(output_dir, filename)
    return path


def do_python_eval(cfg, infos, output_dir, use_07=True):
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    info = {}
    for i, cls in enumerate(cfg.keys()):
        filename = get_voc_results_file_template(output_dir, cls)
        rec, prec, ap = voc_eval(
            filename, infos, cls,
            ovthresh=cfg["base"]["iou_threshold"], use_07_metric=use_07_metric)
        aps += [ap]
        info[cls] = ap

    return np.mean(aps), info


def voc_eval(detpath, infos, classname, ovthresh=0.5, use_07_metric=True):
    recs = {}
    for i, info in enumerate(infos):
        imagename = info['img_id']
        recs[imagename] = info['detail']

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for info in infos:
        imagename = info['img_id']
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap