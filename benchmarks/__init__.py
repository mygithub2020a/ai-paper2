"""
Benchmarking suite for Belavkin Optimizer.

Provides comprehensive benchmark infrastructure for comparing optimizers
on various tasks and datasets.
"""

from .models import ModularArithmeticMLP, ModularCompositionMLP
from .trainer import Trainer, BenchmarkRunner
from .utils import set_seed, save_results, load_results

__all__ = [
    'ModularArithmeticMLP',
    'ModularCompositionMLP',
    'Trainer',
    'BenchmarkRunner',
    'set_seed',
    'save_results',
    'load_results',
]
