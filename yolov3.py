"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: yolov3.py
@time: 2021-12-07 13:09:33
@desc: 
"""
import torch
import argparse
import os
from jjzhk.config import DetectConfig
from lib.yolov3.yolov3_solver import Yolov3Solver


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('-dataroot', default='/Users/JJZHK/data/', type=str, help='')
    parser.add_argument('-model', default='spp', type=str, help='')
    parser.add_argument('-datatype', default='coco', type=str, help='')
    parser.add_argument('-phase', default='test', type=str, help='')
    parser.add_argument('-lr',default=0.001, type=float, help='')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    args.model_type = "YOLOV3"

    config = DetectConfig("cfg")
    config.load_file_list([
        "%s.cfg" % args.datatype,
        "weights.cfg",
        os.path.join(args.model_type, args.datatype, "%s_%s.cfg" % (args.model_type.lower(), args.model))])

    config['dataset']['root'] = os.path.join(args.dataroot, config['dataset']['root'])  # DATA_ROOT
    config['train']['learning_rate'] = args.lr

    config['net']['trained_weights'] = "%s/%s/%s.pth" % (config[args.model_type]["host"], "pretrained", args.model)
    config['net']['test_weights'] = "%s/trained_%s/%s.pth" % (config[args.model_type]["host"],
                                                                        args.datatype,
                                                                        args.model)

    print('model: %s' % args.model)

    solver = Yolov3Solver(config, model_name=args.model)

    if args.phase == 'train':
        solver.train()
    elif args.phase == 'eval':
        solver.eval()
    elif args.phase == 'test':
        solver.test()
    else:
        print(solver.eval_mAP(50)[0])

