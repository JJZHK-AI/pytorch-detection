"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: util.py
@time: 2021-11-10 16:02:39
@desc: 
"""
import numpy as np
import os
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


#region voc
def write_voc_results_file(cfg, output_dir, all_boxes, infos):
    for cls_ind, cls in enumerate(cfg.class_keys()):
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
    for i, cls in enumerate(cfg.class_keys()):
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
#endregion

#region coco
def write_coco_results_file(cfg, all_boxes, res_file,
                            infos, class_to_coco_cat_id):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []

    # class_to_coco_cat_id = dict(zip([c['name'] for c in self.cats],
    #                                 self.dataset.coco.getCatIds()))

    for cls_ind, cls in enumerate(cfg.class_keys()):
        if cls == '__background__':
            continue
        print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                         len(cfg.class_keys()) - 1))
        coco_cat_id = class_to_coco_cat_id[cls]
        results.extend(coco_results_one_category(all_boxes[cls_ind],
                                                 coco_cat_id, infos))
        '''
        if cls_ind ==30:
            res_f = res_file+ '_1.json'
            print('Writing results json to {}'.format(res_f))
            with open(res_f, 'w') as fid:
                json.dump(results, fid)
            results = []
        '''
    # res_f2 = res_file+'_2.json'
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def coco_results_one_category(boxes, cat_id, infos):
    results = []
    for im_ind, info in enumerate(infos):
        dets = np.array(boxes[im_ind]).astype(np.float)
        if list(dets) == []:
            continue
        scores = dets[:, -1]
        xs = dets[:, 0]
        ys = dets[:, 1]
        ws = dets[:, 2] - xs + 1
        hs = dets[:, 3] - ys + 1
        results.extend(
            [{'image_id': info['img_id'],
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results


def do_detection_eval(cfg, annFile, res_file):
    coco = COCO(annFile)
    ann_type = 'bbox'
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, ann_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    return print_detection_eval_metrics(cfg, coco_eval)


def print_detection_eval_metrics(cfg, coco_eval):
    IoU_lo_thresh = cfg['base']['IoU_lo_thresh']
    IoU_hi_thresh = cfg['base']['IoU_hi_thresh']

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    precision = \
        coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])

    mAP = ap_default
    infos = {}
    for cls_ind, cls in enumerate(cfg.class_keys()):
        if cls == '__background__':
            continue
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])

        infos[cls] = ap
    return mAP, infos
#endregion