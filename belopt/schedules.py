import math
from torch.optim.optimizer import Optimizer

class _HyperparameterScheduler:
    def __init__(self, optimizer, param_name, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.param_name = param_name
        self.last_epoch = last_epoch
        self.base_values = [group[param_name] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_name] = value

    def get_values(self):
        raise NotImplementedError

class ConstantScheduler(_HyperparameterScheduler):
    def get_values(self):
        return self.base_values

class CosineAnnealingScheduler(_HyperparameterScheduler):
    def __init__(self, optimizer, param_name, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, param_name, last_epoch)

    def get_values(self):
        if self.last_epoch == 0:
            return self.base_values
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group[self.param_name] + (base_value - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_value, group in
                    zip(self.base_values, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group[self.param_name] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

def get_scheduler(optimizer, param_name, scheduler_name, **kwargs):
    if scheduler_name == 'constant':
        return ConstantScheduler(optimizer, param_name, **kwargs)
    elif scheduler_name == 'cosine':
        return CosineAnnealingScheduler(optimizer, param_name, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
