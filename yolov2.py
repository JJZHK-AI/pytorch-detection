"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov2.py
@time: 2021-11-12 14:13:42
@desc: 
"""
import torch
import argparse
import os
from jjzhk.config import DetectConfig
from lib.yolov2.yolov2_solver import Yolov2Solver


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('-dataroot', default='/Users/JJZHK/data/', type=str, help='')
    parser.add_argument('-model', default='darknet19', type=str, help='')
    parser.add_argument('-datatype', default='coco', type=str, help='')
    parser.add_argument('-phase', default='train', type=str, help='')
    parser.add_argument('-lr',default=0.001, type=float, help='')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    args.imgsize = 448
    config = DetectConfig("cfg")
    config.load_file_list([
        "%s.cfg" % args.datatype,
        "weights.cfg",
        os.path.join("%d" % args.imgsize, "%s" % args.datatype, "yolov2_%s.cfg" % args.model)])

    config['dataset']['root'] = os.path.join(args.dataroot, config['dataset']['root'])  # DATA_ROOT
    config['train']['learning_rate'] = args.lr

    config['net']['trained_weights'] = "%s/%s/%s.pth" % (config["YOLOV2"]["host"], "pretrained", args.model)
    config['net']['test_weights'] = "%s/trained_%s/%s.pth" % (config["YOLOV2"]["host"],
                                                                        args.datatype,
                                                                        args.model)

    print('model: %s, size: %d' % (args.model, args.imgsize))

    solver = Yolov2Solver(config, model_name=args.model)

    if args.phase == 'train':
        solver.train()
    elif args.phase == 'eval':
        solver.eval()
    elif args.phase == 'test':
        solver.test()
    else:
        print(solver.eval_mAP(50)[0])

