"""
Belavkin Optimizer - A novel optimization algorithm derived from quantum filtering equations.

Based on the Belavkin quantum filtering equation, this optimizer implements
a stochastic gradient descent variant with quantum-inspired adaptive damping
and exploration terms.
"""

from .belavkin import BelavkinOptimizer

__all__ = ['BelavkinOptimizer']
__version__ = '0.1.0'
