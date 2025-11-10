"""
Synthetic Tasks for Optimizer Evaluation

This module implements structured learning tasks designed to evaluate
optimizer performance on problems with known phase transitions and
grokking phenomena.

Tasks:
    1. Modular Arithmetic: Learn f(x) = (ax + b) mod p
    2. Modular Composition: Learn f(g(x)) for modular functions
    3. Sparse Parity: Learn k-sparse parity functions

These tasks are chosen because they:
    - Have well-understood learning dynamics
    - Exhibit phase transitions and grokking
    - Allow fine-grained analysis of convergence
    - Are computationally tractable
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import itertools


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic: f(x) = (ax + b) mod p

    Args:
        p (int): Modulus (prime number)
        a (int): Slope parameter
        b (int): Intercept parameter
        train_fraction (float): Fraction of examples for training
        operation (str): Type of operation ('add', 'mult', 'add_mult')
    """

    def __init__(
        self,
        p: int = 97,
        a: int = 1,
        b: int = 0,
        train_fraction: float = 0.5,
        operation: str = 'add',
    ):
        self.p = p
        self.a = a
        self.b = b
        self.operation = operation

        # Generate all possible inputs
        if operation == 'add':
            # Binary operation: (x + y) mod p
            all_pairs = list(itertools.product(range(p), range(p)))
            self.data = torch.tensor(all_pairs, dtype=torch.long)
            self.labels = torch.tensor(
                [(x + y) % p for x, y in all_pairs], dtype=torch.long
            )
        elif operation == 'mult':
            # Binary operation: (x * y) mod p
            all_pairs = list(itertools.product(range(p), range(p)))
            self.data = torch.tensor(all_pairs, dtype=torch.long)
            self.labels = torch.tensor(
                [(x * y) % p for x, y in all_pairs], dtype=torch.long
            )
        elif operation == 'linear':
            # Unary operation: (ax + b) mod p
            self.data = torch.arange(p, dtype=torch.long).unsqueeze(1)
            self.labels = torch.tensor(
                [(a * x + b) % p for x in range(p)], dtype=torch.long
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Split into train/test
        n_total = len(self.data)
        n_train = int(n_total * train_fraction)
        indices = torch.randperm(n_total)
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

    def __getitem__(self, idx):
        if self.is_train:
            real_idx = self.train_indices[idx]
        else:
            real_idx = self.test_indices[idx]

        return self.data[real_idx], self.labels[real_idx]


class ModularCompositionDataset(Dataset):
    """
    Dataset for compositional modular arithmetic: f(g(x))

    Args:
        p (int): Modulus
        train_fraction (float): Fraction for training
    """

    def __init__(self, p: int = 97, train_fraction: float = 0.5):
        self.p = p

        # Generate all inputs
        all_inputs = list(range(p))
        self.data = torch.tensor(all_inputs, dtype=torch.long)

        # f(x) = (2x + 1) mod p, g(x) = (3x + 2) mod p
        # f(g(x)) = (2(3x + 2) + 1) mod p = (6x + 5) mod p
        self.labels = torch.tensor(
            [(6 * x + 5) % p for x in all_inputs], dtype=torch.long
        )

        # Split train/test
        n_total = len(self.data)
        n_train = int(n_total * train_fraction)
        indices = torch.randperm(n_total)
        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]
        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_test(self):
        self.is_train = False

    def __len__(self):
        return len(self.train_indices if self.is_train else self.test_indices)

    def __getitem__(self, idx):
        real_idx = (self.train_indices if self.is_train else self.test_indices)[idx]
        return self.data[real_idx], self.labels[real_idx]


class SparseParityDataset(Dataset):
    """
    Dataset for k-sparse parity functions.

    A k-sparse parity function depends on k out of n input bits.
    Output is XOR of selected bits.

    Args:
        n_bits (int): Total number of input bits
        k_sparse (int): Number of relevant bits
        n_samples (int): Number of samples to generate
        train_fraction (float): Fraction for training
    """

    def __init__(
        self,
        n_bits: int = 10,
        k_sparse: int = 3,
        n_samples: int = 10000,
        train_fraction: float = 0.8,
    ):
        self.n_bits = n_bits
        self.k_sparse = k_sparse

        # Randomly select k relevant indices
        self.relevant_indices = np.random.choice(n_bits, k_sparse, replace=False)

        # Generate random binary inputs
        self.data = torch.randint(0, 2, (n_samples, n_bits), dtype=torch.float32)

        # Compute parity over relevant bits
        relevant_data = self.data[:, self.relevant_indices]
        self.labels = (relevant_data.sum(dim=1) % 2).long()

        # Split train/test
        n_train = int(n_samples * train_fraction)
        indices = torch.randperm(n_samples)
        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]
        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_test(self):
        self.is_train = False

    def __len__(self):
        return len(self.train_indices if self.is_train else self.test_indices)

    def __getitem__(self, idx):
        real_idx = (self.train_indices if self.is_train else self.test_indices)[idx]
        return self.data[real_idx], self.labels[real_idx]


class SimpleMLP(nn.Module):
    """
    Simple multi-layer perceptron for synthetic tasks.

    Args:
        input_dim (int): Input dimension
        hidden_dims (list): List of hidden layer dimensions
        output_dim (int): Output dimension
        activation (str): Activation function ('relu', 'gelu', 'tanh')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu',
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ModularArithmeticModel(nn.Module):
    """
    Model specifically designed for modular arithmetic tasks.

    Uses embedding layer for discrete inputs and predicts over
    modular classes.

    Args:
        p (int): Modulus (vocabulary size)
        hidden_dim (int): Hidden dimension
        n_layers (int): Number of hidden layers
        operation (str): Type of operation ('add', 'mult', 'linear')
    """

    def __init__(
        self,
        p: int = 97,
        hidden_dim: int = 128,
        n_layers: int = 2,
        operation: str = 'add',
    ):
        super().__init__()

        self.p = p
        self.operation = operation

        # Embedding for inputs
        if operation in ['add', 'mult']:
            # Binary operation: embed both inputs
            self.embed_x = nn.Embedding(p, hidden_dim)
            self.embed_y = nn.Embedding(p, hidden_dim)
            input_dim = 2 * hidden_dim
        else:
            # Unary operation
            self.embed_x = nn.Embedding(p, hidden_dim)
            input_dim = hidden_dim

        # Hidden layers
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(hidden_dim, p)

    def forward(self, x):
        if self.operation in ['add', 'mult']:
            # x is [batch, 2] containing (x, y)
            emb_x = self.embed_x(x[:, 0])
            emb_y = self.embed_y(x[:, 1])
            h = torch.cat([emb_x, emb_y], dim=-1)
        else:
            # x is [batch, 1] or [batch]
            if x.dim() == 2:
                x = x.squeeze(1)
            h = self.embed_x(x)

        h = self.mlp(h)
        logits = self.output(h)
        return logits


def create_modular_task(
    p: int = 97,
    operation: str = 'add',
    hidden_dim: int = 128,
    n_layers: int = 2,
    train_fraction: float = 0.5,
    batch_size: int = 512,
) -> Tuple[nn.Module, DataLoader, DataLoader]:
    """
    Create a modular arithmetic task with model and data loaders.

    Args:
        p: Modulus
        operation: Type of operation
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        train_fraction: Fraction for training
        batch_size: Batch size

    Returns:
        model, train_loader, test_loader
    """
    # Create dataset
    dataset = ModularArithmeticDataset(p=p, operation=operation, train_fraction=train_fraction)

    # Create model
    model = ModularArithmeticModel(p=p, hidden_dim=hidden_dim, n_layers=n_layers, operation=operation)

    # Create data loaders
    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return model, train_loader, test_loader


def create_sparse_parity_task(
    n_bits: int = 10,
    k_sparse: int = 3,
    hidden_dims: list = [128, 128],
    n_samples: int = 10000,
    train_fraction: float = 0.8,
    batch_size: int = 128,
) -> Tuple[nn.Module, DataLoader, DataLoader]:
    """
    Create a sparse parity task with model and data loaders.

    Args:
        n_bits: Number of input bits
        k_sparse: Number of relevant bits
        hidden_dims: Hidden layer dimensions
        n_samples: Number of samples
        train_fraction: Fraction for training
        batch_size: Batch size

    Returns:
        model, train_loader, test_loader
    """
    # Create dataset
    dataset = SparseParityDataset(
        n_bits=n_bits, k_sparse=k_sparse, n_samples=n_samples, train_fraction=train_fraction
    )

    # Create model
    model = SimpleMLP(input_dim=n_bits, hidden_dims=hidden_dims, output_dim=2)

    # Create data loaders
    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return model, train_loader, test_loader
