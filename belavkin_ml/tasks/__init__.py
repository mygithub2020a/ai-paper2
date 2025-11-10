"""
Synthetic tasks for optimizer benchmarking.

Includes tasks with known phase transitions and grokking phenomena:
- Modular arithmetic
- Modular composition
- Sparse parity
"""

from belavkin_ml.tasks.modular import (
    ModularArithmeticDataset,
    ModularCompositionDataset,
    create_modular_task,
)
from belavkin_ml.tasks.sparse_parity import (
    SparseParityDataset,
    create_sparse_parity_task,
)

__all__ = [
    'ModularArithmeticDataset',
    'ModularCompositionDataset',
    'create_modular_task',
    'SparseParityDataset',
    'create_sparse_parity_task',
]
