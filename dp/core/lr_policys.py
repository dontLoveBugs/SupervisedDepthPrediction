# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午10:56
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : lr_policys.py

import warnings
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR


class ConstantLR(_LRScheduler):

    def __init__(self, optimizer):
        super(ConstantLR, self).__init__(optimizer)

    def get_lr(self):
        return self.base_lrs


class WarmUpLR(_LRScheduler):
    def __init__(
            self, optimizer, scheduler,
            factor=1.0 / 3,
            iters=500,
            method="linear",
            last_epoch=-1,
    ):
        self.warmup_method = method
        self.scheduler = scheduler
        self.warmup_iters = iters
        self.warmup_factor = factor
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self._step_count < self.warmup_iters:
            if self.warmup_method == "linear":
                alpha = float(self._step_count) / float(self.warmup_iters)
                factor = self.warmup_factor * (1 - alpha) + alpha
            elif self.warmup_method == "constant":
                factor = self.warmup_factor
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.warmup_method))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

        if self._step_count > 0:
            self.scheduler.step(epoch)

        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolyLR(_LRScheduler):

    def __init__(self, optimizer, gamma=0.9, num_iteration=-1):
        self.step_size = num_iteration
        self.gamma = gamma
        super(PolyLR, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self._step_count / self.step_size)**self.gamma for base_lr in self.base_lrs]


__schedulers__ = {
    'step': StepLR,
    'multi_step': MultiStepLR,
    'constant': ConstantLR,
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

    if config.get('warm_up') is not None:
        scheduler = WarmUpLR(optimizer=optimizer, scheduler=scheduler, **config["warm_up"])

    return scheduler
