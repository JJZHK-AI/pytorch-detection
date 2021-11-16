"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: train.py
@time: 2021-11-05 20:56:41
@desc: 
"""
import torch
import argparse
import os
from jjzhk.config import DetectConfig
from lib.yolov1.yolov1_solver import Yolov1Solver

torch.seed(1)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('-dataroot', default='/Users/JJZHK/data/', type=str, help='')
    parser.add_argument('-model', default='resnet50', type=str, help='')
    parser.add_argument('-datatype', default='coco', type=str, help='')
    parser.add_argument('-phase', default='train', type=str, help='')
    # parser.add_argument('-lr',default=0.0005, type=float, help='')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    args.imgsize = 448
    config = DetectConfig("cfg")
    config.load_file_list([
        "%s.cfg" % args.datatype,
        os.path.join("%d" % args.imgsize, "%s" % args.datatype, "yolov1_%s.cfg" % args.model)])

    config['dataset']['root'] = os.path.join(args.dataroot, config['dataset']['root'])  # DATA_ROOT
    # config['base']['backbone'] = args.net
    # config['train']['learning_rate'] = args.lr

    print('model: %s, size: %d' % (args.model, args.imgsize))

    solver = Yolov1Solver(config, model_name=args.model)

    if args.phase == 'train':
        solver.train()
    elif args.phase == 'eval':
        solver.eval()
    else:
        solver.test()





