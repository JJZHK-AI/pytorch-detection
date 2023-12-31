"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov4.py
@time: 2021-10-27 10:24:18
@desc: 
"""
import torch
import argparse
import os
from jjzhk.config import DetectConfig
from lib.yolov4.yolov4_solver import YOLOV4Solver
import torchvision as tv

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('-dataroot', default='/Users/JJZHK/data/', type=str, help='')
    parser.add_argument('-model', default='yolov4', type=str, help='')
    parser.add_argument('-datatype', default='coco', type=str, help='')
    parser.add_argument('-net', default='darknet', type=str, help='')
    parser.add_argument('-phase', default='eval', type=str, help='')
    parser.add_argument('-imgsize', default=608, type=int, help='')
    parser.add_argument('-lr',default=0.001, type=float, help='')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    args.model_type = "YOLOV4"

    config = DetectConfig("cfg")
    config.load_file_list([
        "%s.cfg" % args.datatype,
        "weights.cfg",
        os.path.join(args.model_type, args.datatype, "%s.cfg" % args.net)])
    config.load_backbone_file(os.path.join("backbone", "%s-%s.cfg" % (args.model, args.net)))

    config['dataset']['root'] = os.path.join(args.dataroot, config['dataset']['root'])  # DATA_ROOT
    config['base']['backbone'] = args.net
    config['train']['learning_rate'] = args.lr
    config['net']['test_weights'] = "%s/trained_%s/%s.pth" % (config[args.model.upper()]["host"], args.datatype, args.net)

    print('model: %s, backbone: %s, size: %d' % (args.model, args.net, args.imgsize))

    solver = YOLOV4Solver(config, model_name=args.net)

    if args.phase == 'train':
        solver.train()
    elif args.phase == 'eval':
        solver.eval()
    else:
        solver.test()
