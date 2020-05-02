# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 下午8:12
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : psm_net.py


from dorn.visualizers.base_visualizer import BaseVisualizer
import numpy as np
from dorn.utils.wrappers import tensor2numpy
from dorn.visualizers.utils import disp_to_color, disp_err_to_color, cost_dist_map


class psm_net_visualizer(BaseVisualizer):

    def __init__(self, config, writer=None):
        super(psm_net_visualizer, self).__init__(config, writer)

    def visualize(self, batch, out, epoch=0):
        """
            :param batch_in: minibatch
            :param pred_out: model output for visualization
            :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
            :return: vis_ims: image for visualization.
            """
        fn = batch["fn"]
        lt_ims, rt_ims = batch["left_image_n"].numpy(), batch["right_image_n"].numpy()

        has_gt = False
        if batch.get("left_disparity") is not None:
            disp_gts = tensor2numpy(batch["left_disparity"])
            has_gt = True

        axis = 1 if self.config["horizontally"] else 0
        for i in range(len(fn)):
            lgt_im = lt_ims[i].astype(np.float)
            rgt_im = rt_ims[i].astype(np.float)
            cost_0 = tensor2numpy(out['cost'][0][i])
            disp_0 = tensor2numpy(out['disp'][0][i])
            cost_1 = tensor2numpy(out['cost'][1][i])
            disp_1 = tensor2numpy(out['disp'][1][i])
            cost_2 = tensor2numpy(out['cost'][2][i])
            disp_2 = tensor2numpy(out['disp'][2][i])

            # print("disp 0:", disp_0.shape, "cost 0:", cost_0.shape)
            if has_gt:
                disp_gt = disp_gts[i]

                err0 = disp_err_to_color(disp_0, disp_gt)
                err1 = disp_err_to_color(disp_1, disp_gt)
                err2 = disp_err_to_color(disp_2, disp_gt)

                disp_gt = disp_to_color(disp_gt)

            disp_0 = disp_to_color(disp_0)
            disp_1 = disp_to_color(disp_1)
            disp_2 = disp_to_color(disp_2)

            group = np.concatenate((lgt_im, rgt_im), axis)
            group = np.concatenate((group, disp_0), axis)
            if has_gt:
                group = np.concatenate((group, err0), axis)
            group = np.concatenate((group, disp_1), axis)
            if has_gt:
                group = np.concatenate((group, err1), axis)
            group = np.concatenate((group, disp_2), axis)
            if has_gt:
                group = np.concatenate((group, err2), axis)
                group = np.concatenate((group, disp_gt))

            if self.writer is not None:
                group = group.transpose((2, 0, 1)) / 255.0
                self.writer.add_image(fn[i]+"/image", group, epoch)

            # if self.config.get("cost_x") or self.config.get("cost_y"):
            #     im = lgt_im.copy()

            if self.config.get("cost_x"):
                for x in self.config["cost_x"]:
                    _im = lgt_im.copy()
                    _im[x - 3:x + 3, ::, ::] = [255, 0, 0]
                    cost_x0 = cost_dist_map(cost_0, x)
                    cost_x1 = cost_dist_map(cost_1, x)
                    cost_x2 = cost_dist_map(cost_2, x)
                    # print("# ss:", _im.shape, cost_x0.shape)
                    group = np.concatenate((_im, cost_x0, cost_x1, cost_x2), 0)

                    if self.writer is not None:
                        group = group.transpose((2, 0, 1)) / 255.
                        self.writer.add_image(fn[i]+"/cost_x" + str(x), group, epoch)

            if self.config.get("cost_y"):
                for y in self.config["cost_y"]:
                    _im = lgt_im.copy()
                    _im[::, y - 3:y + 3, ::] = [255, 0, 0]
                    cost_y0 = cost_dist_map(cost_0, y, horizontally=False)
                    cost_y1 = cost_dist_map(cost_1, y, horizontally=False)
                    cost_y2 = cost_dist_map(cost_2, y, horizontally=False)

                    group = np.concatenate((_im, cost_y0, cost_y1, cost_y2), 1)
                    if self.writer is not None:
                        group = group.transpose((2, 0, 1)) / 255.
                        self.writer.add_image(fn[i]+"/cost_y" + str(y), group, epoch)

            # if self.config.get("cost_x") or self.config.get("cost_y"):
            #     im = im.transpose((2, 0, 1)) / 255.
            #     self.writer.add_image(fn[i]+"/cost_xy_", im, epoch)
