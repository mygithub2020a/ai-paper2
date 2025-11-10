"""
Belavkin Optimizer & Belavkin RL

This package implements:
1. BelOpt: A Belavkin-inspired optimizer for deep learning
2. BelRL: A Belavkin-driven RL training scheme

References:
- Belavkin equation: https://en.wikipedia.org/wiki/Belavkin_equation
- Quantum Stochastics and Information (Belavkin & Guta, 2008)
"""

__version__ = "0.1.0"

from .belopt.optim import BelOpt

__all__ = ["BelOpt"]
