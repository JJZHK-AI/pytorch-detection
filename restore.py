"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: restore.py
@time: 2021-11-12 10:39:34
@desc: 
"""
import argparse
import os
import shutil


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('-epoch', default=1, type=int, help='')
    args = parser.parse_args(argv)
    return args


def restore_eval(epoch, folder: str = 'logger'):
    current = epoch
    while True:
        directory = os.path.join(folder, "eval_logs", "%d" % current)
        file = os.path.join(folder, "eval_logs", "%d.txt" % current)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.remove(file)
        else:
            break
        current += 1
    lines = []
    with open(os.path.join(folder, "eval_logs", "eval.log"), 'r') as f:
        lines = f.readlines()
    # epoch 28 - mAP : 0.561
    lines = [x for x in lines if int(x.strip().split(' - ')[0].split(' ')[1]) < epoch]
    os.remove(os.path.join(folder, "eval_logs", "eval.log"))

    with open(os.path.join(folder, "eval_logs", "eval.log"), 'w') as f:
        for line in lines:
            f.write(line)
        f.flush()


def restore_test(epoch: int, folder: str = 'logger'):
    while True:
        directory = os.path.join(folder, "test_logs", "%d" % epoch)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        else:
            break
        epoch += 1


def restore_train(epoch, folder: str = 'logger'):
    current = epoch
    while True:
        file = os.path.join(folder, "train_logs", "%d.pth" % current)
        if os.path.exists(file):
            os.remove(file)
        else:
            break
        current += 1

    lines = []
    with open(os.path.join(folder, "train_logs", "checkpoint.log"), 'r') as f:
        lines = f.readlines()
    # epoch 28 - mAP : 0.561
    lines = [x for x in lines if int(x.strip().split(' - ')[0].split(' ')[1]) < epoch]
    os.remove(os.path.join(folder, "train_logs", "checkpoint.log"))

    with open(os.path.join(folder, "train_logs", "checkpoint.log"), 'w') as f:
        for line in lines:
            f.write(line)
        f.flush()

    with open(os.path.join(folder, "train_logs", "loss.log"), 'r') as f:
        lines = f.readlines()
    # epoch 28 - mAP : 0.561
    lines = [x for x in lines if int(x.strip().split(' - ')[0].split(' ')[1]) < epoch]
    os.remove(os.path.join(folder, "train_logs", "loss.log"))

    with open(os.path.join(folder, "train_logs", "loss.log"), 'w') as f:
        for line in lines:
            f.write(line)
        f.flush()


if __name__ == '__main__':
    args = parse_args()
    epoch = args.epoch

    restore_test(epoch)
    restore_eval(epoch)
    restore_train(epoch)