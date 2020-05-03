#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:33
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""
import numpy as np

from dp.visualizers.utils import depth_to_color, error_to_color
from dp.visualizers.base_visualizer import BaseVisualizer
from dp.utils.wrappers import tensor2numpy


class dorn_visualizer(BaseVisualizer):
    def __init__(self, config, writer=None):
        super(dorn_visualizer, self).__init__(config, writer)

    def visualize(self, batch, out, epoch=0):
        """
            :param batch_in: minibatch
            :param pred_out: model output for visualization, N, 1, H, W
            :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
            :return: vis_ims: image for visualization.
            """
        fn = batch["fn"]
        image = batch["image_n"].numpy()

        has_gt = False
        if batch.get("target") is not None:
            depth_gts = tensor2numpy(batch["target"])
            has_gt = True

        axis = 1 if self.config["horizontally"] else 0
        for i in range(len(fn)):
            image = image[i].astype(np.float)
            depth = tensor2numpy(out['target'][i])[0]
            # print("!! depth shape:", depth.shape)

            if has_gt:
                depth_gt = depth_gts[i]

                err = error_to_color(depth, depth_gt)
                depth_gt = depth_to_color(depth_gt)

            depth = depth_to_color(depth)
            # print("pred:", depth.shape, " target:", depth_gt.shape)
            group = np.concatenate((image, depth), axis)

            if has_gt:
                group = np.concatenate((group, depth_gt), axis)
                group = np.concatenate((group, err), axis)

            if self.writer is not None:
                group = group.transpose((2, 0, 1)) / 255.0
                self.writer.add_image(fn[i] + "/image", group, epoch)