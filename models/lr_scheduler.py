import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, init_lr, total_epochs, last_epoch=-1):
        super(CosineLR, self).__init__(optimizer, last_epoch=-1)
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.total_epochs = total_epochs
        self.last_epoch = last_epoch
        print(f'CosineLR start from epoch(step) {last_epoch} with init_lr {init_lr} ')

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [0.5 * (1 + math.cos(self.last_epoch * math.pi / self.total_epochs)) * self.init_lr for group in
                self.optimizer.param_groups]
