"""
Belavkin Quantum Filtering Framework for Machine Learning

This package implements quantum filtering principles from the Belavkin equation
for machine learning applications, including:
- Track 1: Novel optimization algorithms inspired by quantum filtering
- Track 2: Reinforcement learning using quantum state estimation
"""

__version__ = "0.1.0"

from belavkin_ml.optimizers.belavkin import BelavkinOptimizer
from belavkin_ml.optimizers.belavkin import BelavkinSGD, BelavkinAdam

__all__ = [
    "BelavkinOptimizer",
    "BelavkinSGD",
    "BelavkinAdam",
]
