#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-03 03:31
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : test_resnet.py
"""

import os

os.chdir("../../../")

from dp.modules.backbones.resnet import ResNetBackbone
import unittest


class resnetTestCase(unittest.TestCase):

    def test_load(self):
        backbone = ResNetBackbone(pretrained=True)


if __name__ == "__main__":
    unittest.main()
