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
import torch.utils.data
import numpy as np


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)


@make_iterative_func
def check_nan(vars):
    assert torch.sum(torch.isnan(vars)) == 0., "vas shouldn't have NAN."
    # return torch.sum(torch.isnan(vars)) > 0.


@make_iterative_func
def tensor2cuda(vars):
    assert isinstance(vars, torch.Tensor), "Type of vars must be Torch.tensor"
    return vars.cuda(non_blocking=True)
