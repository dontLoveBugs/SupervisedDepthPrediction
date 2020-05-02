#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 12:11
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def build_metrics(cfg):
    mod = __import__("{}.{}".format(__name__, "metrics"), fromlist=[''])
    return getattr(mod, "StereoMetrics")(max_disp=cfg["model"]["params"]["max_disp"])