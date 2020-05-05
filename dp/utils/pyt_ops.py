#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-03 20:27
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : pyt_ops.py
"""

import torch.nn.functional as F
import numpy as np


def softmax(x, axis=0):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis)
    return softmax_x
