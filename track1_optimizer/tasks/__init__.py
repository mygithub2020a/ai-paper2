"""
Synthetic tasks for testing Belavkin optimizer.

These tasks are designed to exhibit phase transitions and grokking phenomena,
allowing fine-grained analysis of learning dynamics.
"""

from .modular_arithmetic import ModularArithmeticDataset, generate_modular_data
from .modular_composition import ModularCompositionDataset, generate_composition_data
from .sparse_parity import SparseParityDataset, generate_parity_data

__all__ = [
    "ModularArithmeticDataset",
    "generate_modular_data",
    "ModularCompositionDataset",
    "generate_composition_data",
    "SparseParityDataset",
    "generate_parity_data",
]
