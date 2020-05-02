# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 下午4:24
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : metrics.py

import math
import torch

from dp.utils.wrappers import make_nograd_func
from dp.utils.comm import reduce_dict
from dp.metircs.average_meter import AverageMeterList


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


@make_nograd_func
def compute_metric(pred, target):
    valid_mask = target > 0
    pred = pred[valid_mask]
    target = target[valid_mask]

    abs_diff = (pred - target).abs()

    mse = (torch.pow(abs_diff, 2)).mean()
    rmse = torch.sqrt(mse)
    mae = abs_diff.mean()
    absrel = (abs_diff / target).mean()

    d = log10(pred) - log10(target)
    lg10 = d.abs().mean()
    silog = torch.pow(d, 2).mean() - d.mean() * d.mean()

    maxRatio = torch.max(pred / target, target / pred)
    delta1 = (maxRatio < 1.25).float().mean()
    delta2 = (maxRatio < 1.25 ** 2).float().mean()
    delta3 = (maxRatio < 1.25 ** 3).float().mean()

    inv_output = 1 / pred
    inv_target = 1 / target
    abs_inv_diff = (inv_output - inv_target).abs()
    irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    imae = abs_inv_diff.mean()

    metric = dict(mse=mse,
                  rmse=rmse,
                  mae=mae,
                  absrel=absrel,
                  lg10=lg10,
                  silog=silog,
                  delta1=delta1,
                  delta2=delta2,
                  delta3=delta3,
                  irmse=irmse,
                  imae=imae)

    return metric


class Metrics(object):

    def __init__(self, max_disp):
        self.max_disp = max_disp

        # self.irmse = AverageMeterList()
        # self.imae = AverageMeterList()

        self.mse = AverageMeterList()
        self.mae = AverageMeterList()
        self.rmse = AverageMeterList()
        self.absrel = AverageMeterList()

        # self.lg10 = AverageMeterList()
        self.silog = AverageMeterList()

        self.d1 = AverageMeterList()
        self.d2 = AverageMeterList()
        self.d3 = AverageMeterList()

        self.n_stage = -1

    def reset(self):
        # self.irmse.reset()
        # self.imae.reset()

        self.mse.reset()
        self.mae.reset()
        self.rmse.reset()
        self.absrel.reset()

        # self.lg10.reset()
        self.silog.reset()

        self.d1.reset()
        self.d2.reset()
        self.d3.reset()
        self.n_stage = -1

    def compute_metric(self, preds, minibatch):
        gt = minibatch["depth"]
        mask = (gt > 0.) & (gt < self.max_disp)
        if len(gt[mask]) == 0:
            return

        imse, imae, mse, rmse, absrel, lg10, silog, d1, d2, d3 \
            = [], [], [], [], [], [], [], [], [], []

        if self.n_stage == -1:
            self.n_stage = len(preds["depth"])

        for scale_idx in range(self.n_stage):
            pred = preds["depth"][scale_idx]
            metirc_dict = compute_metric(pred, gt)
            metirc_dict = reduce_dict(metirc_dict)
            # imse.append(metirc_dict["imse"])
            # imae.append(metirc_dict["imae"])
            mse.append(metirc_dict["mse"])
            rmse.append(metirc_dict["rmse"])
            absrel.append(metirc_dict["absrel"])
            # lg10.append(metirc_dict["lg10"])
            silog.append(metirc_dict["silog"])
            d1.append(metirc_dict["d1"])
            d2.append(metirc_dict["d2"])
            d3.append(metirc_dict["d3"])

        del imse
        del imae
        del mse
        del rmse
        del absrel
        del lg10
        del silog
        del d1
        del d2
        del d3

    def add_scalar(self, writer=None, tag="Test", epoch=0):
        if writer is None:
            return
        keys = ["stage_{}".format(i) for i in range(self.n_stage)]
        writer.add_scalars(tag + "/mse", dict(zip(keys, self.mse.mean())), epoch)
        writer.add_scalars(tag + "/rmse", dict(zip(keys, self.rmse.mean())))
        writer.add_scalars(tag + "/mae", dict(zip(keys, self.mae.mean())), epoch)
        writer.add_scalars(tag + "/absrml", dict(zip(keys, self.absrel.mean())), epoch)
        writer.add_scalars(tag + "/silog", dict(zip(keys, self.silog.mean())), epoch)
        writer.add_scalars(tag + "/d1", dict(zip(keys, self.d1.mean())), epoch)
        writer.add_scalars(tag + "/d2", dict(zip(keys, self.d2.mean())), epoch)
        writer.add_scalars(tag + "/d3", dict(zip(keys, self.d3.mean())), epoch)

    def get_snapshot_info(self):
        info = "absrel: %.2f" % self.absrel.values()[-1] + "(%.2f)" % self.absrel.mean()[-1]
        info += " rmse: %.2f" % self.rmse.values()[-1] + "(%.2f)" % self.rmse.mean()[-1]
        return info

    def get_result_info(self):
        info = "absrel: %.2f" % self.absrel.mean()[-1] + \
               " rmse: %.2f" % self.rmse.mean()[-1] + \
               " silog: %.2f" % self.silog.mean()[-1]
        return info
