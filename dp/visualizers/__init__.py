# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 下午8:12
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : __init__.py.py


def build_visualizer(cfg, writer=None):
    mod = __import__('{}.{}'.format(__name__, cfg['vis_config']['name']), fromlist=[''])
    return getattr(mod, cfg["vis_config"]["name"] + "_visualizer")(cfg, writer)


