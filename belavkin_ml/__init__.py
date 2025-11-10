"""
Belavkin Quantum Filtering Framework for Machine Learning

This package implements novel machine learning algorithms inspired by
Belavkin's quantum filtering equations.

Tracks:
- Track 1: Belavkin-inspired neural network optimizer
- Track 2: Belavkin framework for deep reinforcement learning
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from belavkin_ml.optimizer import BelavkinOptimizer
from belavkin_ml.optimizer import AdaptiveBelavkinOptimizer

__all__ = [
    "BelavkinOptimizer",
    "AdaptiveBelavkinOptimizer",
]
