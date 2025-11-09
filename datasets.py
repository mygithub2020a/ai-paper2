"""
Synthetic datasets for benchmarking the Belavkin Optimizer.

This module provides modular arithmetic and modular composition tasks
that serve as challenging benchmarks for optimization algorithms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class ModularArithmeticDataset:
    """
    Modular Arithmetic Dataset

    Task: Learn a neural network to compute (a + b) mod p for given inputs a, b
    This is a challenging non-linear regression problem that tests optimization.
    """

    def __init__(
        self,
        modulus: int = 113,
        num_samples: int = 1000,
        seed: int = 42,
        input_dim: int = 2,
    ):
        """
        Args:
            modulus: The modulus p for (a + b) mod p
            num_samples: Number of training samples
            seed: Random seed for reproducibility
            input_dim: Number of input features (should be 2 for a, b)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.modulus = modulus
        self.num_samples = num_samples
        self.input_dim = input_dim

        # Generate random inputs in [0, modulus)
        self.X = torch.randint(0, modulus, (num_samples, input_dim)).float()

        # Compute targets: (sum of inputs) mod modulus
        targets = self.X.sum(dim=1).long()
        self.y = (targets % modulus).float().unsqueeze(1)

        # Normalize inputs and targets
        self.X = self.X / modulus
        self.y = self.y / modulus

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of data."""
        indices = torch.randperm(self.num_samples)[:batch_size]
        return self.X[indices], self.y[indices]

    def get_full_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data."""
        return self.X, self.y


class ModularCompositionDataset:
    """
    Modular Composition Dataset

    Task: Learn a neural network to compute ((a * b) mod p + c) mod p
    This is a more complex task involving both multiplication and addition modulo p.
    """

    def __init__(
        self,
        modulus: int = 113,
        num_samples: int = 1000,
        seed: int = 42,
        input_dim: int = 3,
    ):
        """
        Args:
            modulus: The modulus p
            num_samples: Number of training samples
            seed: Random seed for reproducibility
            input_dim: Number of input features (should be 3 for a, b, c)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.modulus = modulus
        self.num_samples = num_samples
        self.input_dim = input_dim

        # Generate random inputs in [0, modulus)
        self.X = torch.randint(0, modulus, (num_samples, input_dim)).float()

        # Compute targets: ((a * b) mod p + c) mod p
        a = self.X[:, 0].long()
        b = self.X[:, 1].long()
        c = self.X[:, 2].long()

        targets = ((a * b) % self.modulus + c) % self.modulus
        self.y = targets.float().unsqueeze(1)

        # Normalize inputs and targets
        self.X = self.X / modulus
        self.y = self.y / modulus

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of data."""
        indices = torch.randperm(self.num_samples)[:batch_size]
        return self.X[indices], self.y[indices]

    def get_full_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data."""
        return self.X, self.y


class ModularXORDataset:
    """
    Modular XOR Dataset

    Task: Learn (a XOR b) mod p
    This tests the ability to learn logical operations modulo p.
    """

    def __init__(
        self,
        modulus: int = 113,
        num_samples: int = 1000,
        seed: int = 42,
        input_dim: int = 2,
    ):
        """
        Args:
            modulus: The modulus p
            num_samples: Number of training samples
            seed: Random seed for reproducibility
            input_dim: Number of input features (should be 2 for a, b)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.modulus = modulus
        self.num_samples = num_samples
        self.input_dim = input_dim

        # Generate random inputs in [0, modulus)
        self.X = torch.randint(0, modulus, (num_samples, input_dim)).float()

        # Compute targets: (a XOR b) mod modulus
        a = self.X[:, 0].long()
        b = self.X[:, 1].long()
        targets = (a ^ b) % self.modulus
        self.y = targets.float().unsqueeze(1)

        # Normalize inputs and targets
        self.X = self.X / modulus
        self.y = self.y / modulus

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of data."""
        indices = torch.randperm(self.num_samples)[:batch_size]
        return self.X[indices], self.y[indices]

    def get_full_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data."""
        return self.X, self.y


class SimpleNNModel(nn.Module):
    """Simple neural network for regression on synthetic datasets."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Optional[list] = None,
        output_dim: int = 1,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        super(SimpleNNModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        layers = []

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepNNModel(nn.Module):
    """Deeper neural network for more challenging optimization."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Optional[list] = None,
        output_dim: int = 1,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        super(DeepNNModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 64, 64, 32]

        layers = []

        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Add batch normalization for deeper networks
            if i % 2 == 0 and i > 0:
                layers.append(nn.BatchNorm1d(hidden_dim))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_dataset(
    dataset_name: str = "modular_arithmetic",
    modulus: int = 113,
    num_samples: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Factory function to create datasets.

    Args:
        dataset_name: Name of dataset ('modular_arithmetic', 'modular_composition', 'modular_xor')
        modulus: The modulus parameter
        num_samples: Number of samples
        seed: Random seed

    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "modular_arithmetic":
        return ModularArithmeticDataset(
            modulus=modulus, num_samples=num_samples, seed=seed
        )
    elif dataset_name == "modular_composition":
        return ModularCompositionDataset(
            modulus=modulus, num_samples=num_samples, seed=seed
        )
    elif dataset_name == "modular_xor":
        return ModularXORDataset(
            modulus=modulus, num_samples=num_samples, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test datasets
    print("Testing ModularArithmeticDataset...")
    dataset = ModularArithmeticDataset(num_samples=100)
    X, y = dataset.get_full_data()
    print(f"  Input shape: {X.shape}, Output shape: {y.shape}")
    print(f"  Sample input: {X[0]}, Sample output: {y[0]}")

    print("\nTesting ModularCompositionDataset...")
    dataset = ModularCompositionDataset(num_samples=100)
    X, y = dataset.get_full_data()
    print(f"  Input shape: {X.shape}, Output shape: {y.shape}")
    print(f"  Sample input: {X[0]}, Sample output: {y[0]}")

    print("\nTesting ModularXORDataset...")
    dataset = ModularXORDataset(num_samples=100)
    X, y = dataset.get_full_data()
    print(f"  Input shape: {X.shape}, Output shape: {y.shape}")
    print(f"  Sample input: {X[0]}, Sample output: {y[0]}")

    print("\nTesting Models...")
    model = SimpleNNModel(input_dim=2, output_dim=1)
    x = torch.randn(10, 2)
    y = model(x)
    print(f"  SimpleNNModel output shape: {y.shape}")

    model = DeepNNModel(input_dim=2, output_dim=1)
    y = model(x)
    print(f"  DeepNNModel output shape: {y.shape}")
