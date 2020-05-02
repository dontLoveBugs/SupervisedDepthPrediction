# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 下午3:20
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : build.py


import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dp.datasets import _get_dataset


def build_loader(config, is_train=True, world_size=1, distributed=False):
    dataset = _get_dataset(config['data']['name'])(config=config, is_train=is_train)

    sampler = None
    batch_size = config['solver']['batch_size'] if is_train else world_size
    niters_per_epoch = int(np.ceil(dataset.get_length() // batch_size))

    if distributed:
        sampler = DistributedSampler(dataset)
        batch_size = batch_size // world_size

    if is_train:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=10,
                            drop_last=False,
                            shuffle=(sampler is None),
                            pin_memory=False,
                            sampler=sampler)
    else:
        loader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=4,
                            drop_last=False,
                            shuffle=False,
                            pin_memory=False,
                            sampler=sampler)

    return loader, sampler, niters_per_epoch
