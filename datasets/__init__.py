"""
Dataset generators for benchmarking the Belavkin Optimizer.

Provides modular arithmetic and modular composition synthetic datasets
for comprehensive optimizer evaluation.
"""

from .modular_arithmetic import (
    ModularArithmeticDataset,
    generate_modular_addition,
    generate_modular_multiplication,
    generate_modular_subtraction,
)
from .modular_composition import (
    ModularCompositionDataset,
    generate_composition_data,
)

__all__ = [
    'ModularArithmeticDataset',
    'generate_modular_addition',
    'generate_modular_multiplication',
    'generate_modular_subtraction',
    'ModularCompositionDataset',
    'generate_composition_data',
]
