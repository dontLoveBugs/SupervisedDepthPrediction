#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-02 11:42
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : base_visualizer.py
"""


class BaseVisualizer(object):

    def __init__(self, config, writer=None):
        self.config = config["vis_config"]
        self.writer = writer

    def visualize(self, batch, out, epoch=0):
        raise NotImplementedError
