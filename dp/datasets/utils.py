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
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(file), "file not found: {}".format(file)
    from PIL import Image
    img_file = Image.open(file)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), file)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def nomalize(image, imagenet=True):
    if imagenet:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([np.mean(image[:, :, 0]),
                         np.mean(image[:, :, 1]),
                         np.mean(image[:, :, 2])])
        std = np.array([np.std(image[:, :, 0]),
                        np.std(image[:, :, 1]),
                        np.std(image[:, :, 2])])
    image -= mean
    image /= std
    return image
