"""
Modular Composition Task

Task: Learn f(g(x)) where f and g are modular functions
Example: f(x) = (ax + b) mod p, g(x) = (cx + d) mod p
Composite: h(x) = (a(cx + d) + b) mod p

This tests compositional generalization - can the network learn to compose
two modular operations?
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


def generate_composition_data(
    prime: int = 97,
    a: int = 5,
    b: int = 3,
    c: int = 7,
    d: int = 2,
    train_frac: float = 0.5,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate modular composition dataset: h(x) = f(g(x)) where
    f(x) = (ax + b) mod p and g(x) = (cx + d) mod p

    Args:
        prime: Prime modulus p
        a, b: Coefficients for outer function f
        c, d: Coefficients for inner function g
        train_frac: Fraction of data to use for training
        seed: Random seed for train/test split

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Generate all possible inputs
    X = torch.arange(prime).float()

    # Compute g(x) = (cx + d) mod p
    g_x = (c * X + d) % prime

    # Compute f(g(x)) = (a*g(x) + b) mod p
    y = ((a * g_x + b) % prime).long()

    # Create train/test split
    n_train = int(train_frac * prime)

    if seed is not None:
        torch.manual_seed(seed)

    perm = torch.randperm(prime)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X[train_idx].unsqueeze(1)
    y_train = y[train_idx]
    X_test = X[test_idx].unsqueeze(1)
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


class ModularCompositionDataset(Dataset):
    """
    PyTorch Dataset for modular composition tasks.
    """

    def __init__(
        self,
        prime: int = 97,
        a: int = 5,
        b: int = 3,
        c: int = 7,
        d: int = 2,
        train: bool = True,
        train_frac: float = 0.5,
        seed: Optional[int] = 42,
    ):
        """
        Initialize modular composition dataset.

        Args:
            prime: Prime modulus p
            a, b: Coefficients for outer function f
            c, d: Coefficients for inner function g
            train: If True, use training split; else test split
            train_frac: Fraction of data for training
            seed: Random seed
        """
        self.prime = prime
        self.a, self.b = a, b
        self.c, self.d = c, d
        self.train = train

        X_train, y_train, X_test, y_test = generate_composition_data(
            prime=prime, a=a, b=b, c=c, d=d, train_frac=train_frac, seed=seed
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


class CompositionMLP(nn.Module):
    """
    Deeper MLP for modular composition tasks.
    Uses 3-4 layers to allow for learning compositional structure.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 97,
        num_layers: int = 3,
        activation: str = "relu",
    ):
        super(CompositionMLP, self).__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def test_composition_dataset():
    """Test modular composition dataset."""
    print("Testing Modular Composition Dataset...")

    dataset = ModularCompositionDataset(prime=97, train=True, train_frac=0.5)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch shapes: X={x_batch.shape}, y={y_batch.shape}")

    model = CompositionMLP(input_dim=1, hidden_dim=256, output_dim=97, num_layers=3)
    output = model(x_batch)
    print(f"Model output shape: {output.shape}")

    print("✓ Modular composition dataset test passed!\n")


if __name__ == "__main__":
    test_composition_dataset()
