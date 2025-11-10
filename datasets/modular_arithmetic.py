"""
Modular Arithmetic Dataset Generator

Generates synthetic datasets for modular arithmetic operations:
- Modular addition: (a + b) mod p
- Modular multiplication: (a * b) mod p
- Modular subtraction: (a - b) mod p

These tasks test an optimizer's ability to learn discrete algebraic structures.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic operations.

    Args:
        operation (str): Type of operation ('add', 'multiply', 'subtract')
        modulus (int): The modulus p for modular arithmetic
        num_samples (int): Number of samples to generate
        embedding_dim (int): Dimension for embedding integers (default: 32)
        seed (Optional[int]): Random seed for reproducibility
    """

    def __init__(
        self,
        operation: str = 'add',
        modulus: int = 97,
        num_samples: int = 10000,
        embedding_dim: int = 32,
        seed: Optional[int] = None,
    ):
        if operation not in ['add', 'multiply', 'subtract']:
            raise ValueError(f"Invalid operation: {operation}")

        self.operation = operation
        self.modulus = modulus
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate data
        self.data, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate modular arithmetic data."""
        # Sample random integers in [0, modulus)
        a = torch.randint(0, self.modulus, (self.num_samples,))
        b = torch.randint(0, self.modulus, (self.num_samples,))

        # Compute labels based on operation
        if self.operation == 'add':
            labels = (a + b) % self.modulus
        elif self.operation == 'multiply':
            labels = (a * b) % self.modulus
        elif self.operation == 'subtract':
            labels = (a - b) % self.modulus

        # Create embeddings: concatenate one-hot or learned embeddings
        # For simplicity, we'll use a simple encoding scheme
        # Stack a and b as input features
        data = torch.stack([a, b], dim=1).float()

        return data, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def get_embedding_params(self) -> Tuple[int, int]:
        """Returns (vocab_size, embedding_dim) for building models."""
        return self.modulus, self.embedding_dim


def generate_modular_addition(
    modulus: int = 97,
    num_samples: int = 10000,
    train_split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[ModularArithmeticDataset, ModularArithmeticDataset]:
    """
    Generate train and test datasets for modular addition.

    Args:
        modulus: The modulus for modular arithmetic
        num_samples: Total number of samples
        train_split: Fraction of data for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_samples = int(num_samples * train_split)
    test_samples = num_samples - train_samples

    train_ds = ModularArithmeticDataset(
        operation='add',
        modulus=modulus,
        num_samples=train_samples,
        seed=seed,
    )

    test_ds = ModularArithmeticDataset(
        operation='add',
        modulus=modulus,
        num_samples=test_samples,
        seed=seed + 1 if seed is not None else None,
    )

    return train_ds, test_ds


def generate_modular_multiplication(
    modulus: int = 97,
    num_samples: int = 10000,
    train_split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[ModularArithmeticDataset, ModularArithmeticDataset]:
    """Generate train and test datasets for modular multiplication."""
    train_samples = int(num_samples * train_split)
    test_samples = num_samples - train_samples

    train_ds = ModularArithmeticDataset(
        operation='multiply',
        modulus=modulus,
        num_samples=train_samples,
        seed=seed,
    )

    test_ds = ModularArithmeticDataset(
        operation='multiply',
        modulus=modulus,
        num_samples=test_samples,
        seed=seed + 1 if seed is not None else None,
    )

    return train_ds, test_ds


def generate_modular_subtraction(
    modulus: int = 97,
    num_samples: int = 10000,
    train_split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[ModularArithmeticDataset, ModularArithmeticDataset]:
    """Generate train and test datasets for modular subtraction."""
    train_samples = int(num_samples * train_split)
    test_samples = num_samples - train_samples

    train_ds = ModularArithmeticDataset(
        operation='subtract',
        modulus=modulus,
        num_samples=train_samples,
        seed=seed,
    )

    test_ds = ModularArithmeticDataset(
        operation='subtract',
        modulus=modulus,
        num_samples=test_samples,
        seed=seed + 1 if seed is not None else None,
    )

    return train_ds, test_ds


class ModularArithmeticWithEmbedding(Dataset):
    """
    Enhanced version that returns integer indices for embedding lookup.

    This is useful for testing optimizers on discrete input spaces with
    learned embeddings.
    """

    def __init__(
        self,
        operation: str = 'add',
        modulus: int = 97,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ):
        self.operation = operation
        self.modulus = modulus
        self.num_samples = num_samples

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.data, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data as integer indices."""
        a = torch.randint(0, self.modulus, (self.num_samples,))
        b = torch.randint(0, self.modulus, (self.num_samples,))

        if self.operation == 'add':
            labels = (a + b) % self.modulus
        elif self.operation == 'multiply':
            labels = (a * b) % self.modulus
        elif self.operation == 'subtract':
            labels = (a - b) % self.modulus

        # Return as (batch, 2) where each row is [a, b]
        data = torch.stack([a, b], dim=1)
        return data, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
