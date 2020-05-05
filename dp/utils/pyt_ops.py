#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-03 20:27
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : pyt_ops.py
"""

import torch
import torch.nn.functional as F
import numpy as np

from .wrappers import make_iterative_func


def softmax(x, axis=0):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis)
    return softmax_x


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
    assert isinstance(vars, torch.Tensor)
    return torch.sum(torch.isnan(vars)) > 0.
    # assert torch.sum(torch.isnan(vars)) == 0., "vas shouldn't have NAN."
    # return torch.sum(torch.isnan(vars)) > 0.


@make_iterative_func
def tensor2cuda(vars):
    assert isinstance(vars, torch.Tensor), "Type of vars must be Torch.tensor"
    return vars.cuda(non_blocking=True)


@make_iterative_func
def interpolate(vars, size=None, scale_factor=None, mode='nearest', align_corners=None):
    # print("!!!size:", size)
    if not isinstance(vars, torch.Tensor) or vars.dim() < 3:
        return vars
    if vars.dim()==3:
        out = torch.unsqueeze(vars, dim=1)
        out = F.interpolate(out, size, scale_factor, mode, align_corners)
        return out.squeeze(dim=1)
    return F.interpolate(vars, size, scale_factor, mode, align_corners)
