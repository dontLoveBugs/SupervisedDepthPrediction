#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:17
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : ordinal_regression_loss.py
"""

import numpy as np
import torch


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, H, W = gt.shape
        ord_p0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_p0[mask] = 0
        ord_p1 = 1 - ord_p0
        ord_prob = torch.cat((ord_p0, ord_p1), dim=1)
        return ord_prob

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        ord_label = self._create_ord_label(gt)
        loss = prob * ord_label
        return loss.mean()
