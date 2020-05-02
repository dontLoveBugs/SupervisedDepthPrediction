#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-11 16:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : setup.py
"""

from setuptools import setup, find_packages


setup(
        name='dp',
        version='1.0',
        description='Depth Prediction',
        author='wangxin',
        packages=find_packages(exclude=["scripts"])
)