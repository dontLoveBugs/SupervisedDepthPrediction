#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 22:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : Kitti.py
"""

import os
import random
import numpy as np
import cv2


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
        W = int(crop_h / H * W)
        H = crop_h
        # print("w={}, h={}".format(W, H))
        image.resize((W, H), Image.BILINEAR)
        depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        # random crop size
        x = random.randint(0, W - crop_w)
        y = random.randint(0, H - crop_h)
        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[y:y + crop_h, x:x + crop_w]

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, imagenet=self.config['imagenet_normalize'])
        image = image.transpose(2, 0, 1)

        return image, depth, None

    def _te_preprocess(self, image, depth):
        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        W = int(crop_h / H * W)
        H = crop_h
        image.resize((crop_w, H), Image.BILINEAR)
        depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)

        # center crop
        x = (W - crop_w) // 2
        y = (H - crop_h) // 2
        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth =depth[y:y + crop_h, x:x + crop_w]

        # normalize
        image_n = np.array(image).astype(np.float32)
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, imagenet=self.config['imagenet_normalize'])
        image = image.transpose(2, 0, 1)

        output_dict = {"image_n": image_n}

        return image, depth, output_dict
