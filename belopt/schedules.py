import math

class BaseScheduler:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, step):
        raise NotImplementedError

class ConstantScheduler(BaseScheduler):
    def __call__(self, step):
        return self.initial_value

class CosineDecayScheduler(BaseScheduler):
    def __init__(self, initial_value, max_steps):
        super().__init__(initial_value)
        self.max_steps = max_steps

    def __call__(self, step):
        step = min(step, self.max_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.max_steps))
        return self.initial_value * cosine_decay

class InverseSqrtScheduler(BaseScheduler):
    def __init__(self, initial_value, warmup_steps=0):
        super().__init__(initial_value)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_value * (step + 1) / self.warmup_steps
        return self.initial_value / math.sqrt(step + 1)
