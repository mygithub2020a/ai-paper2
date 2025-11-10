"""
Belavkin Reinforcement Learning Framework

Applies quantum filtering principles to reinforcement learning problems,
particularly suited for partially observable environments.

Core components:
- Belief state representation and filtering
- Model-based RL with Belavkin filtering
- Model-free Q-learning variant
- Policy optimization with quantum-inspired updates
"""

from belavkin_ml.rl.belief_state import BelavkinBeliefState
from belavkin_ml.rl.agents import ModelBasedBelavkinAgent, ModelFreeBelavkinAgent

__all__ = [
    "BelavkinBeliefState",
    "ModelBasedBelavkinAgent",
    "ModelFreeBelavkinAgent",
]
