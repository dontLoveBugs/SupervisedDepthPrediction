#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:21
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : pyt_ops.py
"""

from __future__ import print_function, division
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np


def make_iterative_func(func):
    def wrapper(vars, **f_kwargs):
        if isinstance(vars, list):
            return [wrapper(x, **f_kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, **f_kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, **f_kwargs) for k, v in vars.items()}
        else:
            return func(vars, **f_kwargs)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper
