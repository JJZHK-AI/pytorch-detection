"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: layers.py
@time: 2021-09-16 17:44:55
@desc: 
"""
import torch


class ModelLayer:
    def __init__(self, layer_config, **kwargs):
        self._layer_config_ = layer_config
        self._seq_ = layer_config['seq'] if 'seq' in layer_config else 0
        self._modules_list_ = self.transfer_config(**kwargs)

    def transfer_config(self, **kwargs) -> tuple:
        pass

    def layers(self):
        seq = self._layer_config_['seq'] if 'seq' in self._layer_config_ else 0
        if seq == 1:
            seqlist = torch.nn.Sequential()
            for l in self._modules_list_:
                seqlist.add_module(l['name'], l['layer'])
        else:
            seqlist = []
            for l in self._modules_list_:
                seqlist.append(l['layer'])

        return seqlist
