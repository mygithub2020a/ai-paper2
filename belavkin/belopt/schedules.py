"""
Schedulers for learning rate, gamma, and beta in BelOpt.

Provides various schedule types:
- Constant
- Linear decay
- Cosine annealing
- Inverse square root decay
"""

import math
from typing import Optional


class Schedule:
    """Base class for parameter schedules."""

    def __call__(self, step: int) -> float:
        """Get the parameter value at the given step."""
        raise NotImplementedError


class ConstantSchedule(Schedule):
    """Constant parameter value."""

    def __init__(self, value: float):
        """
        Args:
            value: Constant value to return
        """
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value


class LinearSchedule(Schedule):
    """Linear decay from initial to final value."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        """
        Args:
            initial_value: Starting value
            final_value: Ending value
            total_steps: Total number of steps for the schedule
            warmup_steps: Number of warmup steps (linear increase from 0 to initial_value)
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (step / max(1, self.warmup_steps))

        if step >= self.total_steps:
            return self.final_value

        # Linear decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.initial_value + (self.final_value - self.initial_value) * progress


class CosineSchedule(Schedule):
    """Cosine annealing schedule."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        """
        Args:
            initial_value: Starting value
            final_value: Ending value (minimum value)
            total_steps: Total number of steps for the schedule
            warmup_steps: Number of warmup steps (linear increase from 0 to initial_value)
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (step / max(1, self.warmup_steps))

        if step >= self.total_steps:
            return self.final_value

        # Cosine annealing
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_value + (self.initial_value - self.final_value) * cosine_factor


class InverseSqrtSchedule(Schedule):
    """Inverse square root decay schedule."""

    def __init__(
        self,
        initial_value: float,
        warmup_steps: int = 0,
        shift: int = 1,
    ):
        """
        Args:
            initial_value: Initial value (at step=0 or after warmup)
            warmup_steps: Number of warmup steps (linear increase from 0 to initial_value)
            shift: Shift parameter to avoid division by zero (value = initial / sqrt(step + shift))
        """
        self.initial_value = initial_value
        self.warmup_steps = warmup_steps
        self.shift = shift

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (step / max(1, self.warmup_steps))

        # Inverse square root decay
        effective_step = step - self.warmup_steps + self.shift
        return self.initial_value * math.sqrt(self.shift) / math.sqrt(effective_step)


class ExponentialSchedule(Schedule):
    """Exponential decay schedule."""

    def __init__(
        self,
        initial_value: float,
        decay_rate: float,
        decay_steps: int = 1,
        warmup_steps: int = 0,
    ):
        """
        Args:
            initial_value: Initial value
            decay_rate: Decay rate (value *= decay_rate every decay_steps)
            decay_steps: Number of steps between decay applications
            warmup_steps: Number of warmup steps
        """
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (step / max(1, self.warmup_steps))

        # Exponential decay
        effective_step = step - self.warmup_steps
        num_decays = effective_step // self.decay_steps
        return self.initial_value * (self.decay_rate ** num_decays)


class PolynomialSchedule(Schedule):
    """Polynomial decay schedule."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        power: float = 1.0,
        warmup_steps: int = 0,
    ):
        """
        Args:
            initial_value: Starting value
            final_value: Ending value
            total_steps: Total number of steps
            power: Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
            warmup_steps: Number of warmup steps
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.power = power
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (step / max(1, self.warmup_steps))

        if step >= self.total_steps:
            return self.final_value

        # Polynomial decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        decay_factor = (1 - progress) ** self.power
        return self.final_value + (self.initial_value - self.final_value) * decay_factor


class CompositeSchedule(Schedule):
    """Composite schedule that chains multiple schedules."""

    def __init__(self, schedules: list[tuple[int, Schedule]]):
        """
        Args:
            schedules: List of (step_threshold, schedule) tuples.
                      The schedule is used when step >= step_threshold.
                      Should be sorted by step_threshold in ascending order.
        """
        self.schedules = sorted(schedules, key=lambda x: x[0])

    def __call__(self, step: int) -> float:
        for threshold, schedule in reversed(self.schedules):
            if step >= threshold:
                return schedule(step)
        # Fallback to first schedule if step < first threshold
        return self.schedules[0][1](step)


def get_schedule(schedule_type: str, **kwargs) -> Schedule:
    """
    Factory function to create a schedule.

    Args:
        schedule_type: Type of schedule ('constant', 'linear', 'cosine', 'inverse_sqrt', 'exponential', 'polynomial')
        **kwargs: Arguments for the schedule constructor

    Returns:
        Schedule instance
    """
    schedule_map = {
        'constant': ConstantSchedule,
        'linear': LinearSchedule,
        'cosine': CosineSchedule,
        'inverse_sqrt': InverseSqrtSchedule,
        'exponential': ExponentialSchedule,
        'polynomial': PolynomialSchedule,
    }

    if schedule_type not in schedule_map:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Choose from {list(schedule_map.keys())}")

    return schedule_map[schedule_type](**kwargs)
