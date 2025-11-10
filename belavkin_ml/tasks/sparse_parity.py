"""
Sparse parity tasks for optimizer benchmarking.

Sparse parity functions are classic hard learning problems that test
an optimizer's ability to discover sparse, combinatorial structure.

Task: Learn k-sparse parity function
    f(x) = ⊕_{i∈S} x_i  where |S| = k

This is known to be hard for neural networks and exhibits sharp
phase transitions in learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import numpy as np


class SparseParityDataset(Dataset):
    """
    Dataset for k-sparse parity learning.

    Generates binary vectors and computes parity over a random subset of k bits.

    Args:
        n_bits (int): Total number of input bits
        k_sparse (int): Number of bits in parity function
        n_samples (int): Number of samples to generate
        train_fraction (float): Fraction for training
        seed (int): Random seed

    Example:
        >>> dataset = SparseParityDataset(n_bits=10, k_sparse=3, n_samples=1000)
        >>> x, y = dataset[0]
    """

    def __init__(
        self,
        n_bits: int = 10,
        k_sparse: int = 3,
        n_samples: int = 1000,
        train_fraction: float = 0.8,
        seed: int = 42,
        noise_prob: float = 0.0,
    ):
        self.n_bits = n_bits
        self.k_sparse = k_sparse
        self.n_samples = n_samples
        self.noise_prob = noise_prob

        if k_sparse > n_bits:
            raise ValueError(f"k_sparse ({k_sparse}) cannot exceed n_bits ({n_bits})")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Randomly select k bits for parity function
        self.parity_bits = np.random.choice(n_bits, size=k_sparse, replace=False)
        self.parity_bits = sorted(self.parity_bits)

        # Generate random binary inputs
        self.inputs = torch.randint(0, 2, (n_samples, n_bits), dtype=torch.float32)

        # Compute parity labels
        parity_inputs = self.inputs[:, self.parity_bits]
        self.labels = (parity_inputs.sum(dim=1) % 2).long()

        # Add label noise if specified
        if noise_prob > 0:
            noise_mask = torch.rand(n_samples) < noise_prob
            self.labels[noise_mask] = 1 - self.labels[noise_mask]

        # Split into train/test
        indices = torch.randperm(n_samples)
        n_train = int(n_samples * train_fraction)

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]
        self.is_train = True

    def train(self):
        """Set to training mode."""
        self.is_train = True
        return self

    def test(self):
        """Set to test mode."""
        self.is_train = False
        return self

    def __len__(self):
        if self.is_train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx):
        if self.is_train:
            actual_idx = self.train_indices[idx]
        else:
            actual_idx = self.test_indices[idx]

        x = self.inputs[actual_idx]
        y = self.labels[actual_idx]

        return x, y

    def get_true_parity_bits(self) -> List[int]:
        """Return the true indices of the parity bits."""
        return self.parity_bits.tolist()


class ParityMLP(nn.Module):
    """
    MLP for sparse parity learning.

    Args:
        n_bits (int): Number of input bits
        hidden_dims (list): List of hidden layer dimensions
        activation (str): Activation function
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        n_bits: int,
        hidden_dims: list = [128],
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_bits = n_bits

        layers = []
        prev_dim = n_bits

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_sparse_parity_task(
    n_bits: int = 10,
    k_sparse: int = 3,
    n_samples: int = 1000,
    hidden_dims: list = [128],
    train_fraction: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    noise_prob: float = 0.0,
) -> Tuple[nn.Module, DataLoader, DataLoader, SparseParityDataset]:
    """
    Create a complete sparse parity task.

    Args:
        n_bits (int): Number of input bits
        k_sparse (int): Sparsity (number of bits in parity)
        n_samples (int): Total number of samples
        hidden_dims (list): Hidden layer dimensions
        train_fraction (float): Fraction for training
        batch_size (int): Batch size
        seed (int): Random seed
        noise_prob (float): Label noise probability

    Returns:
        model, train_loader, test_loader, dataset

    Example:
        >>> model, train_dl, test_dl, ds = create_sparse_parity_task(n_bits=10, k_sparse=3)
        >>> print(f"True parity bits: {ds.get_true_parity_bits()}")
    """
    # Create dataset
    dataset = SparseParityDataset(
        n_bits=n_bits,
        k_sparse=k_sparse,
        n_samples=n_samples,
        train_fraction=train_fraction,
        seed=seed,
        noise_prob=noise_prob,
    )

    # Create model
    model = ParityMLP(
        n_bits=n_bits,
        hidden_dims=hidden_dims,
    )

    # Create separate dataset objects for train/test
    train_dataset = SparseParityDataset(
        n_bits=n_bits, k_sparse=k_sparse, n_samples=n_samples,
        train_fraction=train_fraction, seed=seed, noise_prob=noise_prob
    )
    train_dataset.parity_bits = dataset.parity_bits
    train_dataset.inputs = dataset.inputs
    train_dataset.labels = dataset.labels
    train_dataset.train_indices = dataset.train_indices
    train_dataset.test_indices = dataset.test_indices
    train_dataset.train()

    test_dataset = SparseParityDataset(
        n_bits=n_bits, k_sparse=k_sparse, n_samples=n_samples,
        train_fraction=train_fraction, seed=seed, noise_prob=noise_prob
    )
    test_dataset.parity_bits = dataset.parity_bits
    test_dataset.inputs = dataset.inputs
    test_dataset.labels = dataset.labels
    test_dataset.train_indices = dataset.train_indices
    test_dataset.test_indices = dataset.test_indices
    test_dataset.test()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return model, train_loader, test_loader, dataset


def evaluate_parity_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Evaluate parity model.

    Args:
        model: Neural network
        dataloader: Test dataloader
        device: Device

    Returns:
        accuracy, loss
    """
    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += len(y)
            total_loss += loss.item() * len(y)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return accuracy, avg_loss


def compute_bit_importance(
    model: nn.Module,
    dataset: SparseParityDataset,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Compute importance score for each input bit using gradient-based attribution.

    This helps verify if the model has learned the correct sparse structure.

    Args:
        model: Trained model
        dataset: Parity dataset
        device: Device

    Returns:
        importance_scores: Tensor of shape [n_bits] with importance per bit
    """
    model.eval()
    model.to(device)

    # Use integrated gradients or simple gradient magnitude
    importance = torch.zeros(dataset.n_bits)

    for x, y in [(dataset.inputs[i:i+1], dataset.labels[i:i+1])
                 for i in range(min(100, len(dataset)))]:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        if x.grad is not None:
            importance += x.grad.abs().sum(dim=0).cpu()

    importance = importance / min(100, len(dataset))

    return importance
