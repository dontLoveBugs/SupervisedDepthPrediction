#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 21:06
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""

import numpy as np
import torch
import torch.nn as nn

from dp.modules.backbones.resnet import ResNetBackbone
from dp.modules.encoders.KITTISceneModule import SceneUnderstandingModule as KittiSceneModule
from dp.modules.encoders.NYUSceneModule import SceneUnderstandingModule as NyuSceneModule
from dp.modules.decoders.OrdinalRegression import OrdinalRegressionLayer
from dp.modules.losses.ordinal_regression_loss import OrdinalRegressionLoss


class DepthPredModel(nn.Module):

    def __init__(self, ord_num, gamma=1.0, beta=80.0, scene="kitti", discretization="SID"):
        super().__init__()
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization
        self.backbone = ResNetBackbone()
        if scene == "kitti":
            self.SceneUnderstandingModule = KittiSceneModule()
        else:
            self.SceneUnderstandingModule = NyuSceneModule()
        self.regression_layer = OrdinalRegressionLayer()
        self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)

    def forward(self, image, target=None):
        feat = self.backbone(input)
        feat = self.SceneUnderstandingModule(feat)
        if self.training:
            prob = self.regression_layer(feat)
            loss = self.criterion(prob, target)
            return loss

        prob, label = self.regression_layer(feat)
        if self.discretization == "SID":
            t_0 = torch.exp(np.log(self.beta)*label/self.ord_num)
            t_1 = torch.exp(np.log(self.beta)*(label+1)/self.ord_num)
            depth = (t_0+t_1)/2-self.gamma
        else:
            t_0 = 1.0+(self.beta-1.0)*label/self.ord_num
            t_1 = 1.0+(self.beta-1.0)*(label+1)/self.ord_num
            depth = (t_0+t_1)/2-self.gamma
        return {"depth": depth, "prob": prob, "label": label}
