import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealing(_LRScheduler):
    """
    A cosine annealing scheduler that can be applied to any
    hyperparameter in the optimizer's param_groups.
    """
    def __init__(self, optimizer, param_name, T_max, eta_min=0, last_epoch=-1):
        self.param_name = param_name
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealing, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        In this context, `get_lr` is a bit of a misnomer since we're scheduling
        any hyperparameter, not just the learning rate. We'll use it to calculate
        the new value for the hyperparameter we're scheduling.
        """
        if self.last_epoch == 0:
            return [group[self.param_name] for group in self.optimizer.param_groups]

        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group[self.param_name] + (base_val - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_val, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]

        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group[self.param_name] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """
        This is the closed-form solution for the learning rate at a given epoch.
        We'll adapt it to schedule our target hyperparameter.
        """
        return [self.eta_min + (base_val - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_val in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, val in zip(self.optimizer.param_groups, self._get_closed_form_lr()):
            param_group[self.param_name] = val
