# -*- coding: utf-8 -*-
# @Time    : 2020/1/1 下午8:08
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : prob.py

import os
import numpy as np
from PIL import Image


def PILLoader(file):
    return Image.open(file).convert('RGB')


def KittiDepthLoader(file):
    # loads depth map D from png file
    assert os.path.exists(file), "file not found: {}".format(file)
    depth_png = np.array(Image.open(file), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


def nomalize(image, type="mean"):
    assert type in ["mean", "norm", "imagenet-mean", "imagenet-norm"], \
        "normlize type only support 'mean', 'norm', 'imagenet-mean', 'imagenet-norm', but not '{}'".format(type)
    if type == 'mean':
        mean = np.array([np.mean(image[:, :, 0]),
                         np.mean(image[:, :, 1]),
                         np.mean(image[:, :, 2])])
        return image-mean
    if type == 'imagenet-mean':
        mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        return image-mean
    if type == 'norm':
        mean = np.array([np.mean(image[:, :, 0]),
                         np.mean(image[:, :, 1]),
                         np.mean(image[:, :, 2])])
        std = np.array([np.std(image[:, :, 0]),
                        np.std(image[:, :, 1]),
                        np.std(image[:, :, 2])])
        image -= mean
        image /= std
        return image
    if type == 'imagenet-norm':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image -= mean
        image /= std
        return image
    raise NotImplementedError

