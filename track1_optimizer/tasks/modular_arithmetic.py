"""
Modular Arithmetic Task

Task: Learn f(x) = (ax + b) mod p for various primes p

This task exhibits grokking behavior where models suddenly transition from
memorization to generalization. The discrete, structured nature makes it
ideal for studying optimization dynamics.

References:
    - Power et al. (2022). "Grokking: Generalization beyond overfitting on small algorithmic datasets"
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


def generate_modular_data(
    prime: int = 97,
    a: int = 5,
    b: int = 3,
    train_frac: float = 0.5,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate modular arithmetic dataset: f(x) = (ax + b) mod p

    Args:
        prime: Prime modulus p
        a: Multiplicative coefficient
        b: Additive coefficient
        train_frac: Fraction of data to use for training
        seed: Random seed for train/test split

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Generate all possible inputs
    X = torch.arange(prime).float()

    # Compute outputs: f(x) = (ax + b) mod p
    y = ((a * X + b) % prime).long()

    # Create train/test split
    n_train = int(train_frac * prime)

    if seed is not None:
        torch.manual_seed(seed)

    perm = torch.randperm(prime)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X[train_idx].unsqueeze(1)  # Shape: (n_train, 1)
    y_train = y[train_idx]
    X_test = X[test_idx].unsqueeze(1)  # Shape: (n_test, 1)
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


class ModularArithmeticDataset(Dataset):
    """
    PyTorch Dataset for modular arithmetic tasks.

    Example:
        >>> dataset = ModularArithmeticDataset(prime=97, train=True)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for x, y in loader:
        ...     # x: input values, y: target class labels
        ...     pass
    """

    def __init__(
        self,
        prime: int = 97,
        a: int = 5,
        b: int = 3,
        train: bool = True,
        train_frac: float = 0.5,
        seed: Optional[int] = 42,
    ):
        """
        Initialize modular arithmetic dataset.

        Args:
            prime: Prime modulus p (default: 97)
            a: Multiplicative coefficient (default: 5)
            b: Additive coefficient (default: 3)
            train: If True, use training split; else test split
            train_frac: Fraction of data for training (default: 0.5)
            seed: Random seed (default: 42)
        """
        self.prime = prime
        self.a = a
        self.b = b
        self.train = train

        X_train, y_train, X_test, y_test = generate_modular_data(
            prime=prime, a=a, b=b, train_frac=train_frac, seed=seed
        )

        if train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ModularMLP(nn.Module):
    """
    Simple MLP for modular arithmetic tasks.

    Architecture: Input -> Hidden Layer(s) -> Output (classification)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 97,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension (1 for scalar inputs)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes = prime)
            num_layers: Number of hidden layers
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super(ModularMLP, self).__init__()

        # Select activation
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1)

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)


def test_modular_dataset():
    """Quick test of modular arithmetic dataset."""
    print("Testing Modular Arithmetic Dataset...")

    dataset = ModularArithmeticDataset(prime=97, train=True, train_frac=0.5)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")

    # Test with DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch shapes: X={x_batch.shape}, y={y_batch.shape}")

    # Test model
    model = ModularMLP(input_dim=1, hidden_dim=128, output_dim=97)
    output = model(x_batch)
    print(f"Model output shape: {output.shape}")

    print("✓ Modular arithmetic dataset test passed!\n")


if __name__ == "__main__":
    test_modular_dataset()
