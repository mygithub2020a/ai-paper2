"""Synthetic datasets for benchmarking optimizers."""

from .mod_arith import (
    ModularAddition,
    ModularMultiplication,
    ModularInverse,
    ModularPower,
    generate_modular_dataset,
)
from .mod_comp import ModularComposition

__all__ = [
    "ModularAddition",
    "ModularMultiplication",
    "ModularInverse",
    "ModularPower",
    "ModularComposition",
    "generate_modular_dataset",
]
