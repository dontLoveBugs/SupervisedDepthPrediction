#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 22:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : Kitti.py
"""

import os
import cv2
import math
import random
import numpy as np


from PIL import Image

from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import nomalize, PILLoader, KittiDepthLoader


class Kitti(BaseDataset):

    def __init__(self, config, is_train=True, image_loader=PILLoader, depth_loader=KittiDepthLoader):
        super().__init__(config, is_train, image_loader, depth_loader)
        file_list = "./dp/datasets/lists/kitti_{}.list".format(self.split)
        with open(file_list, "r") as f:
            self.filenames = f.readlines()

    def _parse_path(self, index):
        image_path, depth_path = self.filenames[index].split()
        image_path = os.path.join(self.root, image_path)
        depth_path = os.path.join(self.root, depth_path)
        return image_path, depth_path

    def _tr_preprocess(self, image, depth):
        crop_h, crop_w = self.config["tr_crop_size"]
        # resize
        W, H = image.size
        dH, dW = depth.shape

        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        # scale_h, scale_w = max(crop_h/H, 1.0), max(crop_w/W, 1.0)
        # scale = max(scale_h, scale_w)
        # H, W = math.ceil(scale*H), math.ceil(scale*W)
        # H, W = max(int(scale*H), crop_h), max(int(scale*W), crop_w)

        # print("w={}, h={}".format(W, H))
        scale = max(crop_h / H, 1.0)
        H, W = max(crop_h, H), math.ceil(scale * W)
        image = image.resize((W, H), Image.BILINEAR)
        # depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)
        # print("image shape:", image.size, " depth shape:", depth.shape)

        crop_dh, crop_dw = int(crop_h / scale), int(crop_w / scale)

        # random crop size
        x = random.randint(0, W - crop_w)
        y = random.randint(0, H - crop_h)
        dx, dy = math.floor(x/scale), math.floor(y/scale)
        # print("corp dh = {}, crop dw = {}".format(crop_dh, crop_dw))

        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[dy:dy + crop_dh, dx:dx + crop_dw]
        # print("depth shape: ", depth.shape)

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        return image, depth, None

    def _te_preprocess(self, image, depth):
        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        dH, dW = depth.shape

        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        # scale_h, scale_w = max(crop_h/H, 1.0), max(crop_w/W, 1.0)
        # scale = max(scale_h, scale_w)
        # # H, W = math.ceil(scale*H), math.ceil(scale*W)
        scale = max(crop_h / H, 1.0)
        H, W = max(crop_h, H), math.ceil(scale * W)
        # H, W = max(int(scale*H), crop_h), max(int(scale*W), crop_w)

        image_n = image.copy()
        image = image.resize((W, H), Image.BILINEAR)
        crop_dh, crop_dw = int(crop_h/scale), int(crop_w/scale)
        # print("corp dh = {}, crop dw = {}".format(crop_dh, crop_dw))
        # depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        # center crop
        x = (W - crop_w) // 2
        y = (H - crop_h) // 2
        dx = (dW - crop_dw) // 2
        dy = (dH - crop_dh) // 2

        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[dy:dy + crop_dh, dx:dx + crop_dw]
        image_n = image_n.crop((dx, dy, dx + crop_dw, dy + crop_dh))

        # normalize
        image_n = np.array(image_n).astype(np.float32)
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        output_dict = {"image_n": image_n}

        return image, depth, output_dict
