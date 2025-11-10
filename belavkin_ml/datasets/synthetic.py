"""
Synthetic datasets for optimizer evaluation.

These datasets are designed to test specific learning dynamics:
1. Modular Arithmetic: Phase transitions and grokking phenomena
2. Modular Composition: Compositional generalization
3. Sparse Parity: Sample complexity analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import itertools


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic tasks: f(x) = (ax + b) mod p

    This task exhibits grokking phenomena and phase transitions in learning.

    Args:
        p: Prime modulus (default: 97)
        a: Multiplier (default: random)
        b: Offset (default: random)
        train_fraction: Fraction of examples for training (default: 0.5)
        operation: Type of operation: 'linear', 'addition', 'multiplication', 'division'
            (default: 'addition')
        seed: Random seed for reproducibility

    Example:
        >>> dataset = ModularArithmeticDataset(p=97, operation='addition')
        >>> train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        p: int = 97,
        a: Optional[int] = None,
        b: Optional[int] = None,
        train_fraction: float = 0.5,
        operation: str = 'addition',
        seed: int = 42,
    ):
        self.p = p
        self.operation = operation
        self.train_fraction = train_fraction
        self.seed = seed

        np.random.seed(seed)

        # Generate all possible examples
        if operation == 'linear':
            # f(x) = (ax + b) mod p
            self.a = a if a is not None else np.random.randint(1, p)
            self.b = b if b is not None else np.random.randint(0, p)

            self.inputs = np.arange(p)
            self.targets = (self.a * self.inputs + self.b) % p

        elif operation == 'addition':
            # f(x, y) = (x + y) mod p
            pairs = list(itertools.product(range(p), range(p)))
            self.inputs = np.array(pairs)
            self.targets = (self.inputs[:, 0] + self.inputs[:, 1]) % p

        elif operation == 'multiplication':
            # f(x, y) = (x * y) mod p
            pairs = list(itertools.product(range(p), range(p)))
            self.inputs = np.array(pairs)
            self.targets = (self.inputs[:, 0] * self.inputs[:, 1]) % p

        elif operation == 'division':
            # f(x, y) = (x / y) mod p (using modular inverse)
            pairs = [(x, y) for x in range(p) for y in range(1, p)]
            self.inputs = np.array(pairs)
            # Compute modular division using Fermat's little theorem: y^(-1) = y^(p-2) mod p
            targets = []
            for x, y in pairs:
                y_inv = pow(y, p - 2, p)
                targets.append((x * y_inv) % p)
            self.targets = np.array(targets)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Split into train and test
        n_train = int(len(self.inputs) * train_fraction)
        indices = np.random.permutation(len(self.inputs))

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]

        self.is_train = True

    def set_train(self):
        """Set to training mode."""
        self.is_train = True

    def set_test(self):
        """Set to test mode."""
        self.is_train = False

    def __len__(self):
        if self.is_train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.train_indices if self.is_train else self.test_indices
        actual_idx = indices[idx]

        x = self.inputs[actual_idx]
        y = self.targets[actual_idx]

        # Convert to tensors
        if self.operation == 'linear':
            x_tensor = torch.tensor([x], dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)

        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

    def get_info(self) -> dict:
        """Returns dataset statistics."""
        return {
            'p': self.p,
            'operation': self.operation,
            'total_examples': len(self.inputs),
            'train_examples': len(self.train_indices),
            'test_examples': len(self.test_indices),
            'input_dim': 1 if self.operation == 'linear' else 2,
            'output_dim': self.p,
        }


class ModularCompositionDataset(Dataset):
    """
    Dataset for modular composition tasks: f(g(x)) where f, g are modular functions.

    Tests compositional generalization: can the model learn to compose functions?

    Args:
        p: Prime modulus (default: 97)
        f_type: Type of function f: 'linear', 'quadratic' (default: 'linear')
        g_type: Type of function g: 'linear', 'quadratic' (default: 'linear')
        train_fraction: Fraction for training (default: 0.5)
        seed: Random seed

    Example:
        >>> dataset = ModularCompositionDataset(p=97, f_type='linear', g_type='linear')
    """

    def __init__(
        self,
        p: int = 97,
        f_type: str = 'linear',
        g_type: str = 'linear',
        train_fraction: float = 0.5,
        seed: int = 42,
    ):
        self.p = p
        self.f_type = f_type
        self.g_type = g_type
        self.train_fraction = train_fraction

        np.random.seed(seed)

        # Generate random function parameters
        self.f_params = self._generate_function_params(f_type)
        self.g_params = self._generate_function_params(g_type)

        # Generate all inputs
        self.inputs = np.arange(p)

        # Compute targets: f(g(x))
        g_outputs = self._apply_function(self.inputs, g_type, self.g_params)
        self.targets = self._apply_function(g_outputs, f_type, self.f_params)

        # Train/test split
        n_train = int(len(self.inputs) * train_fraction)
        indices = np.random.permutation(len(self.inputs))

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]

        self.is_train = True

    def _generate_function_params(self, func_type: str) -> dict:
        """Generate random parameters for a function."""
        if func_type == 'linear':
            return {
                'a': np.random.randint(1, self.p),
                'b': np.random.randint(0, self.p),
            }
        elif func_type == 'quadratic':
            return {
                'a': np.random.randint(1, self.p),
                'b': np.random.randint(0, self.p),
                'c': np.random.randint(0, self.p),
            }
        else:
            raise ValueError(f"Unknown function type: {func_type}")

    def _apply_function(self, x: np.ndarray, func_type: str, params: dict) -> np.ndarray:
        """Apply a modular function to input."""
        if func_type == 'linear':
            return (params['a'] * x + params['b']) % self.p
        elif func_type == 'quadratic':
            return (params['a'] * x**2 + params['b'] * x + params['c']) % self.p
        else:
            raise ValueError(f"Unknown function type: {func_type}")

    def set_train(self):
        self.is_train = True

    def set_test(self):
        self.is_train = False

    def __len__(self):
        if self.is_train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.train_indices if self.is_train else self.test_indices
        actual_idx = indices[idx]

        x = self.inputs[actual_idx]
        y = self.targets[actual_idx]

        x_tensor = torch.tensor([x], dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

    def get_info(self) -> dict:
        return {
            'p': self.p,
            'f_type': self.f_type,
            'g_type': self.g_type,
            'total_examples': len(self.inputs),
            'train_examples': len(self.train_indices),
            'test_examples': len(self.test_indices),
        }


class SparseParityDataset(Dataset):
    """
    Dataset for k-sparse parity functions.

    Tests sample complexity: how many examples needed to learn parity of k bits?

    Args:
        n_bits: Total number of input bits (default: 10)
        k_sparse: Number of relevant bits (default: 3)
        n_examples: Number of examples to generate (default: 1000)
        train_fraction: Fraction for training (default: 0.8)
        seed: Random seed

    Example:
        >>> dataset = SparseParityDataset(n_bits=10, k_sparse=3, n_examples=1000)
    """

    def __init__(
        self,
        n_bits: int = 10,
        k_sparse: int = 3,
        n_examples: int = 1000,
        train_fraction: float = 0.8,
        seed: int = 42,
    ):
        self.n_bits = n_bits
        self.k_sparse = k_sparse
        self.n_examples = n_examples
        self.train_fraction = train_fraction

        np.random.seed(seed)

        # Randomly select k relevant bit positions
        self.relevant_bits = np.random.choice(n_bits, k_sparse, replace=False)

        # Generate random binary inputs
        self.inputs = np.random.randint(0, 2, size=(n_examples, n_bits))

        # Compute targets: XOR of relevant bits
        relevant_values = self.inputs[:, self.relevant_bits]
        self.targets = np.sum(relevant_values, axis=1) % 2

        # Train/test split
        n_train = int(n_examples * train_fraction)
        indices = np.random.permutation(n_examples)

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]

        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_test(self):
        self.is_train = False

    def __len__(self):
        if self.is_train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.train_indices if self.is_train else self.test_indices
        actual_idx = indices[idx]

        x = self.inputs[actual_idx]
        y = self.targets[actual_idx]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

    def get_info(self) -> dict:
        return {
            'n_bits': self.n_bits,
            'k_sparse': self.k_sparse,
            'relevant_bits': self.relevant_bits.tolist(),
            'total_examples': self.n_examples,
            'train_examples': len(self.train_indices),
            'test_examples': len(self.test_indices),
        }


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders from a dataset.

    Args:
        dataset: Dataset object with set_train() and set_test() methods
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        train_loader, test_loader
    """
    dataset.set_train()
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    dataset.set_test()
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
