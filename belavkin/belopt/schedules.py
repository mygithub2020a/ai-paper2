import math

class _BaseScheduler:
    """Base class for hyperparameter schedulers."""
    def __init__(self, optimizer, param_name):
        self.optimizer = optimizer
        self.param_name = param_name
        self.base_values = [group[param_name] for group in optimizer.param_groups]
        self.last_step = 0

    def get_value(self):
        raise NotImplementedError("Subclasses must implement get_value()")

    def step(self):
        self.last_step += 1
        values = self.get_value()
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_name] = value

class ConstantScheduler(_BaseScheduler):
    """A scheduler that keeps the hyperparameter constant."""
    def get_value(self):
        return self.base_values

class InverseSquareRootDecay(_BaseScheduler):
    """
    Decays the hyperparameter with an inverse square root schedule.
    value = base_value / sqrt(step)
    """
    def __init__(self, optimizer, param_name, warmup_steps=0):
        super().__init__(optimizer, param_name)
        self.warmup_steps = warmup_steps

    def get_value(self):
        if self.last_step < self.warmup_steps:
             # Linear warmup
            return [base_value * (self.last_step / self.warmup_steps) for base_value in self.base_values]
        if self.last_step == 0:
            return self.base_values

        decay_factor = 1.0 / math.sqrt(self.last_step)
        return [base_value * decay_factor for base_value in self.base_values]

class CosineDecay(_BaseScheduler):
    """
    Decays the hyperparameter with a cosine schedule.
    value = base_value * 0.5 * (1 + cos(pi * step / total_steps))
    """
    def __init__(self, optimizer, param_name, total_steps, warmup_steps=0):
        super().__init__(optimizer, param_name)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_value(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup
            return [base_value * (self.last_step / self.warmup_steps) for base_value in self.base_values]

        if self.last_step >= self.total_steps:
            return [0.0] * len(self.base_values)

        progress = (self.last_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_value * cosine_decay for base_value in self.base_values]
