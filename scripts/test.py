#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-03 05:16
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : test.py
"""


from PIL import Image


img = Image.open("demo_01.png")
img.show()
print(img.size)

import numpy as np

x = np.array(img)[:, :, 0]
print(x.shape)

imgx=Image.fromarray(x)
print(imgx.size)
imgx.show()