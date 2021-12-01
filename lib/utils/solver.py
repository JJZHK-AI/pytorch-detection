"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: solver.py
@time: 2021-10-27 11:24:10
@desc: 
"""
import os
import torch
from jjzhk.logger import AILogger
from jjzhk.device import device
import lib.dataset as d
from jjzhk.progressbar import ProgressBar
from jjzhk.config import DetectConfig
from lib.utils.util import do_python_eval
import torch.utils.model_zoo


class Solver(object):
    def __init__(self, cfg:DetectConfig, model_name):
        super(Solver, self).__init__()
        self.cfg = cfg
        self._train_loader_ = None
        self._eval_loader_ = None
        self._test_loader_ = None
        self.phase = None
        self.model_urls = {
        'vgg16'       : 'https://github.com/JJZHK/AIModels/releases/download/1.0/vgg16_SDC.pth',
        'resnet50'    : 'https://github.com/JJZHK/AIModels/releases/download/1.0/resnet50_SDC.pth',
        'resnet152'   : 'https://github.com/JJZHK/AIModels/releases/download/1.0/resnet152_SDC.pth',
        'darknet19'   : 'https://github.com/JJZHK/AIModels/releases/download/1.0/darknet19_SDC.pth',
        'darknet53'   : 'https://github.com/JJZHK/AIModels/releases/download/1.0/darknet53_SDC.pth',
        'mobilenetv1' : 'https://github.com/JJZHK/AIModels/releases/download/1.0/mobilenetv1_SDC.pth',
        'mobilenetv2' : 'https://github.com/JJZHK/AIModels/releases/download/1.0/mobilenetv2_SDC.pth',
    }
        self.logger = AILogger(output="logger")
        self.barCfg = {'bar': 'halloween', 'spinner': None, 'receipt_text': False, 'monitor': True,
                       'stats': True, 'elapsed': False}
        self._train_path_ = os.path.join("logger", "train_logs")
        self._eval_path_ = os.path.join("logger", "eval_logs")
        self._test_path_ = os.path.join("logger", "test_logs")
        self.checkpoint_file = os.path.join(self._train_path_, "checkpoint.log")
        self.loss_file = os.path.join(self._train_path_, "loss.log")
        self.eval_file = os.path.join(self._eval_path_, "eval.log")
        # self.train_path, self.eval_path, \
        # self.test_path, self.checkpoint_file, \
        # self.loss_file, self.eval_file = self.logger.get_path_files()
        '''
        logger/train_logs
        logger/eval_logs
        logger/test_logs
        logger/train_logs/checkpoint.txt
        logger/train_logs/loss.txt
        logger/eval_logs/eval.txt
        '''
        self.model = self._get_model_(model_name)
        self.model.to(device)

        self._init_others_()

    #region private
    def _get_model_(self, model_name) -> torch.nn.Module:
        return self.get_model(model_name)

    def _init_others_(self):
        return self.init_others()

    def _resume_checkpoint_(self, weights, justInitBase=False):
        return self._load_check_point_(weights, justInitBase)

    def _test_epoch_(self, epoch, model):
        if self._test_loader_ is None:
            self._test_dataset_ = self._init_dataset_('test')
            self._test_loader_ = self.init_test_loader()

        model.eval()

        output = os.path.join(self._test_path_, "%d" % epoch)
        if not os.path.exists(output):
            os.mkdir(output)

        self.test_epoch(epoch, model)

    def _eval_epoch_(self, epoch, model):
        if self._eval_loader_ is None:
            self._eval_dataset_ = self._init_dataset_('eval')
            self._eval_loader_ = self.init_eval_loader()

        model.eval()
        if not os.path.exists(os.path.join(self._eval_path_, str(epoch))):
            os.mkdir(os.path.join(self._eval_path_, str(epoch)))

        self.eval_epoch(epoch, model)

    def _init_dataset_(self, phase) -> object:
        print('init %s dataset' % phase)
        return d.get_dataset(self.cfg, phase)

    def _get_optimizer_(self):
        parameters = self.get_train_parameters()
        opt_lower = self.cfg['train']['optimizer'].lower()

        if opt_lower == 'sgd':
            optimizer = torch.optim.SGD(
                parameters, lr=float(self.cfg['train']['learning_rate']),
                momentum=float(self.cfg['train']['momentum']),
                weight_decay=float(self.cfg['train']['weight_decay']))
        elif opt_lower == 'adam':
            optimizer = torch.optim.Adam(
                parameters, lr=float(self.cfg['train']['learning_rate']),
                eps=float(self.cfg['train']['epsilon']),
                betas=(0.9, 0.999),
                weight_decay=float(self.cfg['train']['weight_decay']))
        elif opt_lower == 'adadelta':
            optimizer = torch.optim.Adadelta(
                parameters, lr=float(self.cfg['train']['learning_rate']),
                eps=float(self.cfg['train']['epsilon']),
                weight_decay=float(self.cfg['train']['weight_decay']))
        elif opt_lower == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                parameters, lr=float(self.cfg['train']['learning_rate']),
                alpha=0.9, eps=float(self.cfg['train']['epsilon']),
                momentum=float(self.cfg['train']['momentum']),
                weight_decay=float(self.cfg['train']['weight_decay']))
        elif opt_lower == 'adamw':
            optimizer = torch.optim.AdamW(parameters, float(self.cfg['train']['learning_rate']))
        else:
            raise ValueError("Expected optimizer method in [sgd, adam, adadelta, rmsprop], but received "
                             "{}".format(opt_lower))

        return optimizer

    def _get_loss_(self):
        return self.get_loss() # self.cfg, priors=self.priors if hasattr(self, 'priors') else None

    def _get_scheduler_(self):
        optimizer = self._optimizer_
        scheduler = None

        if self.cfg['train']['scheduler'] == 'step':
            raise Exception('not implement')
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size=self.cfg.TRAIN.LR_SCHEDULER.STEPS[0],
            #                                             gamma=self.cfg.TRAIN.LR_SCHEDULER.GAMMA)
        elif self.cfg['train']['scheduler'] == 'multi_step':
            raise Exception('not implement')
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            #                                                  milestones=self.cfg.TRAIN.LR_SCHEDULER.STEPS,
            #                                                  gamma=self.cfg.TRAIN.LR_SCHEDULER.GAMMA)
        elif self.cfg['train']['scheduler'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=self.cfg['train']['gamma'])
        elif self.cfg['train']['scheduler'] == 'SGDR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg['train']['max_epochs'])
        elif self.cfg['train']['scheduler'] == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.burnin_schedule)
        elif self.cfg['train']['scheduler'] == 'reducelr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler

    def _init_train_(self) -> int:
        previous = self._find_previous_()
        if previous:
            start_epoch = previous[0][-1]
            self._resume_checkpoint_(previous[1][-1])
        else:
            start_epoch = self._init_weights_()

        return start_epoch

    def _find_previous_(self):
        if not os.path.exists(self.checkpoint_file):
            return False

        with open(self.checkpoint_file, 'r') as f:
            lineList = f.readlines()

        if lineList == []:
            return False

        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find('-')])
            checkpoint = line[line.find('-') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def _init_weights_(self):
        self._resume_checkpoint_('', justInitBase=True)
        return 0

    def _load_check_point_(self, weights, justInitBase=False):
        if justInitBase: # 训练时加载pretrained weights
            self._load_backbone_weights_()
        else: # 加载一个已经训练好的weights
            print("loading weights from %s" % weights)
            if 'https://' in weights: # 已经训练好的最终结果，一般用来做eval或者test
                checkpoint = torch.utils.model_zoo.load_url(weights, map_location=device)
                self.model.load_init_weights(checkpoint)
            else: # 还未上传的训练结果，一般用来做某一epoch的weights以继续训练
                self.model.load_init_weights(torch.load(weights), map_location=device)

    def _load_backbone_weights_(self):
        print("loading init weights from %s" % self.cfg['net']['trained_weights'])
        weights = torch.utils.model_zoo.load_url(self.cfg['net']['trained_weights'])
        self.model.load_backbone_weights(weights)

    def _find_previous_eval(self):
        if not os.path.exists(self.eval_file):
            return False

        with open(self.eval_file, 'r') as f:
            lineList = f.readlines()

        if lineList == []:
            return False

        epoches, mAPs = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find('-')])
            mAP = line[line.find('-') + 2:-1]
            epoches.append(epoch)
            mAPs.append(mAP)
        return epoches, mAPs
    #endregion

    #region virtual
    def get_model(self, model_name) -> torch.nn.Module:
        pass

    def init_others(self):
        pass

    def init_test_loader(self) -> object:
        pass

    def init_eval_loader(self) -> object:
        pass

    def init_train_loader(self) -> object:
        pass

    def test_epoch(self, epoch, model):
        pass

    def eval_epoch(self, epoch, model):
        pass

    def get_train_parameters(self) -> list:
        pass

    def get_loss(self):
        pass

    def before_train(self):
        pass

    def change_lr(self, max_epochs, current_epoch, lr) -> float:
        pass

    def train_epoch(self, epoch, bar, newir) -> tuple:
        pass
    #endregion

    #region public
    def train(self):
        self.phase = "train"
        if self._train_loader_ is None:
            self._train_dataset_ = self._init_dataset_('train')
            self._train_loader_ = self.init_train_loader()
            self._optimizer_ = self._get_optimizer_()
            self._criterion_ = self._get_loss_()
            if self._criterion_ is not None:
                self._criterion_.to(device)
            self._lr_scheduler_ = self._get_scheduler_()

        print("start training...")
        start_epoch = self._init_train_()
        newir = float(self.cfg['train']['learning_rate'])
        evalIter = self.cfg["train"]["evaliter"]
        testIter = self.cfg["train"]["testiter"]

        previous_eval = self._find_previous_eval()
        if previous_eval:
            eval_epoch = previous_eval[0][-1]
        else:
            eval_epoch = 0

        if evalIter is not None and evalIter != 0 and start_epoch > eval_epoch\
                and start_epoch % evalIter == 0:
            self._eval_epoch_(start_epoch, self.model)

        if testIter is not None and testIter != 0 and start_epoch % testIter == 0\
                and start_epoch > 0\
                and not os.path.exists(os.path.join(self._test_path_, "%d" % start_epoch)):
            self._test_epoch_(start_epoch, self.model)
        max_epochs = self.cfg['train']['max_epochs']
        bar = ProgressBar(max_epochs, len(self._train_loader_), "Loss:%.3f;AvgLoss:%.3f;LR:%.6f")

        self.before_train()
        for epoch in range(start_epoch + 1, max_epochs + 1):
            self.model.train()

            newir = self.change_lr(max_epochs, epoch, newir)

            avg_loss_per_epoch, time = self.train_epoch(epoch, bar, newir)

            resume_checkpoints = {
                'state_dict': self.model.module.state_dict() if hasattr(self.model,
                                                                        'module') else self.model.state_dict()
                # 'lr_scheduler': self._lr_scheduler_.state_dict(),
                # 'optimizer': self._optimizer_.state_dict()
            }
            self.logger.save_checkpoints_file(epoch, resume_checkpoints)
            self.logger.save_loss(epoch, avg_loss_per_epoch, newir, time)

            if evalIter is not None and evalIter != 0 and epoch % evalIter == 0:
                self._eval_epoch_(epoch, self.model)

            if testIter is not None and testIter != 0 and epoch % testIter == 0:
                self._test_epoch_(epoch, self.model)

    def eval(self):
        self.phase = "eval"
        self._resume_checkpoint_(self.cfg['net']['test_weights'])
        self._eval_epoch_(0, self.model)

    def test(self):
        self.phase = "test"
        self._resume_checkpoint_(self.cfg['net']['test_weights'])
        self._test_epoch_(0, self.model)

    def eval_mAP(self, epoch):
        if self._eval_loader_ is None:
            self._eval_dataset_ = self._init_dataset_('eval')
            self._eval_loader_ = self.init_eval_loader()

        infos = []
        bar = ProgressBar(1, len(self._eval_loader_), "")
        for i, sampler in enumerate(self._eval_loader_):
            images, info = sampler[0], sampler[2]
            infos.append(info[0])
            bar.show(1)

        return do_python_eval(self.cfg, infos, os.path.join(self._eval_path_, "%d" % epoch))


    #endregion