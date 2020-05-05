# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 下午7:05
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : pyt_io.py

import os
import time
import logging
from collections import OrderedDict

import torch
if torch.__version__ >= "1.2.0":
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_model(model, model_file, distributed=False, device=torch.device('cpu')):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if distributed:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logging.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logging.info(
        "Load model, Time usage: IO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))


def create_summary_writer(logdir=None):
    # assert os.path.exists(logdir), 'Log file dir is not existed.'
    ensure_dir(logdir)

    log_path = os.path.join(logdir, 'tensorboard')
    # if os.path.isdir(log_path):
    #     shutil.rmtree(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = SummaryWriter(log_path)
    return logger