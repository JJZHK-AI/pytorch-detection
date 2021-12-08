"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: ssd.py
@time: 2021-10-27 10:24:10
@desc: 
"""
import torch
import argparse
import os
from jjzhk.config import DetectConfig
from lib.ssd.ssd_solver import SSDSolver


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('-dataroot', default='/Users/JJZHK/data/', type=str, help='')
    parser.add_argument('-model', default='ssd', type=str, help='')
    parser.add_argument('-datatype', default='voc', type=str, help='')
    parser.add_argument('-net', default='vgg16', type=str, help='')
    parser.add_argument('-phase', default='test', type=str, help='')
    parser.add_argument('-imgsize', default=300, type=int, help='')
    parser.add_argument('-lr',default=0.001, type=float, help='')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    config = DetectConfig("cfg")
    config.load_file_list([
        "%s.cfg" % args.datatype,
        "weights.cfg",
        os.path.join(args.model.upper(), args.datatype, "%s_%d.cfg" % (args.net, args.imgsize))])

    config['dataset']['root'] = os.path.join(args.dataroot, config['dataset']['root'])  # DATA_ROOT
    config['train']['learning_rate'] = args.lr
    config['net']['trained_weights'] = "%s/%s/%s.pth" % (config["SSD"]["host"], "pretrained", args.net)
    config['net']['test_weights'] = "%s/trained_%s/%s_%s_%d.pth" % (config["SSD"]["host"], args.datatype,
                                                                    args.model, args.net, args.imgsize)

    print('model: %s, backbone: %s, size: %d' % (args.model, args.net, args.imgsize))

    solver = SSDSolver(config, model_name=args.model)

    if args.phase == 'train':
        solver.train()
    elif args.phase == 'eval':
        solver.eval()
    else:
        solver.test()