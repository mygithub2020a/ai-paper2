"""
Sparse Parity Task

Task: Learn k-sparse parity functions
Given binary input x ∈ {0,1}^n, output is XOR of k selected bits.

This is a classic hard learning problem that tests:
- Sample complexity
- Ability to discover sparse structure
- Boolean circuit learning

Example: n=10, k=3, indices=[2,5,7]
    Input: [1,0,1,0,0,1,0,1,0,1]
    Output: x[2] ⊕ x[5] ⊕ x[7] = 1 ⊕ 1 ⊕ 1 = 1
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List


def generate_parity_data(
    n_bits: int = 10,
    k_sparse: int = 3,
    n_samples: int = 1000,
    train_frac: float = 0.7,
    seed: Optional[int] = 42,
    parity_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Generate sparse parity dataset.

    Args:
        n_bits: Input dimension
        k_sparse: Number of bits involved in parity
        n_samples: Total number of samples to generate
        train_frac: Fraction of data for training
        seed: Random seed
        parity_indices: Specific indices for parity (if None, randomly select)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, parity_indices)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Select k indices for parity
    if parity_indices is None:
        parity_indices = sorted(np.random.choice(n_bits, k_sparse, replace=False).tolist())

    # Generate random binary inputs
    X = torch.randint(0, 2, (n_samples, n_bits)).float()

    # Compute parity: XOR of selected bits
    y = torch.zeros(n_samples).long()
    for idx in parity_indices:
        y ^= X[:, idx].long()

    # Create train/test split
    n_train = int(train_frac * n_samples)

    perm = torch.randperm(n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test, parity_indices


class SparseParityDataset(Dataset):
    """
    PyTorch Dataset for sparse parity tasks.
    """

    def __init__(
        self,
        n_bits: int = 10,
        k_sparse: int = 3,
        n_samples: int = 1000,
        train: bool = True,
        train_frac: float = 0.7,
        seed: Optional[int] = 42,
        parity_indices: Optional[List[int]] = None,
    ):
        """
        Initialize sparse parity dataset.

        Args:
            n_bits: Input dimension
            k_sparse: Number of bits in parity
            n_samples: Number of samples
            train: If True, use training split; else test split
            train_frac: Fraction for training
            seed: Random seed
            parity_indices: Specific parity indices (optional)
        """
        self.n_bits = n_bits
        self.k_sparse = k_sparse
        self.train = train

        X_train, y_train, X_test, y_test, self.parity_indices = generate_parity_data(
            n_bits=n_bits,
            k_sparse=k_sparse,
            n_samples=n_samples,
            train_frac=train_frac,
            seed=seed,
            parity_indices=parity_indices,
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


class ParityMLP(nn.Module):
    """
    MLP for sparse parity learning.

    Uses 2-layer architecture as specified in the proposal.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 2,  # Binary classification
        activation: str = "relu",
    ):
        super(ParityMLP, self).__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def test_parity_dataset():
    """Test sparse parity dataset."""
    print("Testing Sparse Parity Dataset...")

    dataset = SparseParityDataset(n_bits=10, k_sparse=3, n_samples=1000, train=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Parity indices: {dataset.parity_indices}")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch shapes: X={x_batch.shape}, y={y_batch.shape}")

    model = ParityMLP(input_dim=10, hidden_dim=128, output_dim=2)
    output = model(x_batch)
    print(f"Model output shape: {output.shape}")

    # Verify parity computation
    x_sample = dataset.X[0]
    y_sample = dataset.y[0]
    manual_parity = 0
    for idx in dataset.parity_indices:
        manual_parity ^= int(x_sample[idx].item())
    print(f"Sample parity check: computed={y_sample.item()}, manual={manual_parity}")
    assert y_sample.item() == manual_parity, "Parity computation error!"

    print("✓ Sparse parity dataset test passed!\n")


if __name__ == "__main__":
    test_parity_dataset()
