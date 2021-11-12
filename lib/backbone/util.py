"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: util.py
@time: 2021-09-16 13:38:09
@desc: 
"""
import torch
from jjzhk.config import DetectConfig
from lib.backbone.layer_zoo import get_layer


def create_modules(module_defs, cfg: DetectConfig, func=None):
    mdef_net = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = [] #torch.nn.ModuleList()
    routs = []
    filters = 0
    mdef_summary = {}
    for i, mdef in enumerate(module_defs):
        mdef['index'] = i
        mdef['inFilters'] = output_filters[-1]
        mdef['filter_list'] = output_filters
        type = mdef['type']

        mdef_summary[type] = mdef_summary[type] + 1 if type in mdef_summary else 0

        if type == 'convolutional':
            filters = mdef['filters']
            l = get_layer(mdef, routs = routs, mlist = module_list, mdefsummary = mdef_summary)
            if isinstance(l, torch.nn.Sequential):
                module_list.append(l)
            elif isinstance(l, list):
                [module_list.append(lay) for lay in l]
        elif type == 'maxpool':
            l = get_layer(mdef, routs = routs, mlist = module_list, mdefsummary = mdef_summary)
            if isinstance(l, list):
                [module_list.append(lay) for lay in l]
            else:
                module_list.append(l)
        elif func != None:
            l, filters = func(mdef, cfg, routs = routs, mlist = module_list, mdefsummary = mdef_summary)
            if isinstance(l, torch.nn.Sequential):
                module_list.append(l)
            elif isinstance(l, list):
                [module_list.append(lay) for lay in l]
            else:
                module_list.append(l)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary, mdef_summary


def layer_to_config(name, layer):
    return {
        "name": name,
        "layer": layer
    }
