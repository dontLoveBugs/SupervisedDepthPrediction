#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 17:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : oridinal_regression_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        A = x[:, ::2, :, :]
        B = x[:, 1::2, :, :]

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)
        concat_feats = torch.cat((A, B), dim=1).contiguous()

        if self.training:
            ord_prob = F.log_softmax(concat_feats, dim=1)
            return ord_prob.view(-1, ord_num, H, W)

        ord_prob = F.softmax(C, dim=1)[:, 1, ::]
        ord_prob = ord_prob.view(-1, ord_num, H, W)
        ord_label = torch.sum((ord_prob > 0.5), dim=1).view(-1, 1, H, W)
        return ord_prob, ord_label

