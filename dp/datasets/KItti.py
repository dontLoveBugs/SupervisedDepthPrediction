#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 22:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : eigen_kitti.py
"""

import os
import random
import numpy as np

from PIL import Image

from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import nomalize


class KittiSet(BaseDataset):

    def __init__(self, config):
        super().__init__(config)
        file_list = "kitti_{}".format(self.split)
        with open(file_list, "r") as f:
            self.filenames = f.readlines()
        # if self.split == "train" or self.split == "trainval":
        #     self.preprocess = self._tr_preprocess
        # else:
        #     self.preprocess = self._te_preprocess

    def _parse_path(self, index):
        if self.split:
            return os.path.join(self.root, self.filenames[index])
        image_path, depth_path = self.filenames[index].split(" ")
        image_path = os.path.join(self.root, image_path)
        depth_path = os.path.join(self.root, depth_path)
        return image_path, depth_path

    def _tr_preprocess(self, image, depth):
        crop_h, crop_w = self.config["tr_crop_size"]
        # resize
        W, H = image.size
        H = int(crop_w / W * H)
        image.resize((crop_w, H), Image.BILINEAR)

        # random crop size
        x = random.randint(0, W - crop_w)
        y = random.randint(0, H - crop_h)
        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth = depth[y:y + crop_h, x:x + crop_w]

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image = nomalize(image, imagenet=self.config['data']['imagenet_normalize'])
        image = image.transpose(2, 0, 1)

        return image, depth, None

    def _te_preprocess(self, image, depth):
        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        H = int(crop_w / W * H)
        image.resize((crop_w, H), Image.BILINEAR)

        # center crop
        x = (W - crop_w) // 2
        y = (H - crop_h) // 2
        image = image.crop((x, y, x + crop_w, y + crop_h))
        depth =depth[y:y + crop_h, x:x + crop_w]

        # normalize
        image = np.asarray(image).astype(np.float32) / 255.0
        image_n = np.copy(image)
        image = nomalize(image, imagenet=self.config['data']['imagenet_normalize'])
        image = image.transpose(2, 0, 1)

        output_dict = {"image_n": image_n}

        return image, depth, output_dict
