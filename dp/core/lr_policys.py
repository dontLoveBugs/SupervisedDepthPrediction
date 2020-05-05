# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午10:56
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : lr_policys.py


from .lr_schedulers import StepLR, MultiStepLR, ConstantLR, PolyLR, WarmUpLR


__schedulers__ = {
    'step': StepLR,
    'multi_step': MultiStepLR,
    'constant': ConstantLR,
    'poly': PolyLR
}


def _get_lr_policy(config, optimizer):
    if config['name'] not in __schedulers__:
        raise NotImplementedError

    if config['name'] == 'step':
        scheduler = __schedulers__[config['name']](
            optimizer=optimizer,
            step_size=config['params']['step_size'],
            gamma=config['params']['gamma']
        )

    if config['name'] == 'multi_step':
        scheduler = __schedulers__[config['name']](
            optimizer=optimizer,
            milestones=config['params']['milestones'],
            gamma=config['params']['gamma']
        )

    if config['name'] == 'constant':
        scheduler = __schedulers__[config['name']](optimizer)

    if config['name'] == 'poly':
        scheduler = __schedulers__[config['name']](optimizer,
                                                   gamma=config["params"]["gamma"],
                                                   n_iteration=config["params"]["n_iteration"])

    if config.get('warm_up') is not None:
        scheduler = WarmUpLR(optimizer=optimizer, scheduler=scheduler, **config["warm_up"])

    return scheduler
