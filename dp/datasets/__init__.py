#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:46
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def _get_dataset(name):
    mod = __import__('{}.{}'.format(__name__, name), fromlist=[''])
    return getattr(mod, name)
