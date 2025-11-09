import math

def constant_schedule(initial_value):
    """Returns a function that always returns the initial value."""
    def schedule(step):
        return initial_value
    return schedule

def cosine_decay_schedule(initial_value, decay_steps, min_value=0.0):
    """Returns a function that implements cosine decay."""
    def schedule(step):
        if step >= decay_steps:
            return min_value
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        decayed_value = (initial_value - min_value) * cosine_decay + min_value
        return decayed_value
    return schedule

def inverse_sqrt_decay_schedule(initial_value, warmup_steps=0):
    """Returns a function that implements inverse square root decay."""
    def schedule(step):
        if step < warmup_steps:
            return initial_value * (step + 1) / warmup_steps
        return initial_value * math.sqrt(warmup_steps) / math.sqrt(step + 1)
    return schedule
