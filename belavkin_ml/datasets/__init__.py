"""
Datasets for Belavkin optimizer evaluation.

Includes:
- Synthetic tasks: Modular arithmetic, composition, sparse parity
- Standard benchmarks: MNIST, CIFAR-10
- RL environments: Gridworld, control tasks
"""

from belavkin_ml.datasets.synthetic import (
    ModularArithmeticDataset,
    ModularCompositionDataset,
    SparseParityDataset,
)

__all__ = [
    "ModularArithmeticDataset",
    "ModularCompositionDataset",
    "SparseParityDataset",
]
