"""
Modular arithmetic tasks for optimizer benchmarking.

These tasks exhibit grokking phenomena and are useful for studying
optimization dynamics and generalization.

Tasks:
1. Modular arithmetic: f(x) = (ax + b) mod p
2. Modular composition: f(g(x)) where f, g are modular functions
3. Modular division: f(x, y) = (x / y) mod p
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
import numpy as np


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic tasks.

    Generates examples of the form:
        f(x) = (ax + b) mod p

    where p is a prime number and a, b are coefficients.

    Args:
        p (int): Prime modulus
        a (int): Multiplicative coefficient
        b (int): Additive coefficient
        train_fraction (float): Fraction of data for training
        seed (int): Random seed for train/test split

    Example:
        >>> dataset = ModularArithmeticDataset(p=97, a=3, b=5)
        >>> x, y = dataset[0]
    """

    def __init__(
        self,
        p: int = 97,
        a: int = 1,
        b: int = 0,
        train_fraction: float = 0.5,
        seed: int = 42,
    ):
        self.p = p
        self.a = a % p
        self.b = b % p
        self.train_fraction = train_fraction

        # Generate all possible inputs
        self.all_inputs = torch.arange(p)

        # Compute labels: f(x) = (ax + b) mod p
        self.all_labels = (self.a * self.all_inputs + self.b) % p

        # Split into train/test
        np.random.seed(seed)
        indices = np.random.permutation(p)
        n_train = int(p * train_fraction)

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

        x = self.all_inputs[actual_idx]
        y = self.all_labels[actual_idx]

        # One-hot encode input and output
        x_onehot = torch.zeros(self.p)
        x_onehot[x] = 1.0

        return x_onehot, y.long()


class ModularCompositionDataset(Dataset):
    """
    Dataset for modular function composition.

    Learns f(g(x)) where:
        g(x) = (a1*x + b1) mod p
        f(y) = (a2*y + b2) mod p

    This tests compositional generalization and deeper learning dynamics.

    Args:
        p (int): Prime modulus
        a1, b1 (int): Coefficients for g(x)
        a2, b2 (int): Coefficients for f(y)
        train_fraction (float): Fraction of data for training
        seed (int): Random seed
    """

    def __init__(
        self,
        p: int = 97,
        a1: int = 2,
        b1: int = 1,
        a2: int = 3,
        b2: int = 2,
        train_fraction: float = 0.5,
        seed: int = 42,
    ):
        self.p = p
        self.a1, self.b1 = a1 % p, b1 % p
        self.a2, self.b2 = a2 % p, b2 % p
        self.train_fraction = train_fraction

        # Generate all possible inputs
        self.all_inputs = torch.arange(p)

        # Compute composition: f(g(x))
        g_x = (self.a1 * self.all_inputs + self.b1) % p
        self.all_labels = (self.a2 * g_x + self.b2) % p

        # Split into train/test
        np.random.seed(seed)
        indices = np.random.permutation(p)
        n_train = int(p * train_fraction)

        self.train_indices = indices[:n_train]
        self.test_indices = indices[n_train:]
        self.is_train = True

    def train(self):
        self.is_train = True
        return self

    def test(self):
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

        x = self.all_inputs[actual_idx]
        y = self.all_labels[actual_idx]

        # One-hot encode input
        x_onehot = torch.zeros(self.p)
        x_onehot[x] = 1.0

        return x_onehot, y.long()


class ModularMLP(nn.Module):
    """
    MLP for modular arithmetic tasks.

    Args:
        input_dim (int): Input dimension (typically equal to modulus p)
        output_dim (int): Output dimension (typically equal to modulus p)
        hidden_dims (list): List of hidden layer dimensions
        activation (str): Activation function ('relu', 'gelu', 'tanh')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512],
        activation: str = 'relu',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_modular_task(
    task_type: str = 'arithmetic',
    p: int = 97,
    hidden_dims: list = [512],
    train_fraction: float = 0.5,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[nn.Module, DataLoader, DataLoader]:
    """
    Create a complete modular task with model and dataloaders.

    Args:
        task_type (str): 'arithmetic' or 'composition'
        p (int): Prime modulus
        hidden_dims (list): Hidden layer dimensions
        train_fraction (float): Fraction for training
        batch_size (int): Batch size
        seed (int): Random seed

    Returns:
        model, train_loader, test_loader

    Example:
        >>> model, train_loader, test_loader = create_modular_task(p=97)
    """
    # Create dataset
    if task_type == 'arithmetic':
        dataset = ModularArithmeticDataset(
            p=p,
            a=np.random.randint(1, p),
            b=np.random.randint(0, p),
            train_fraction=train_fraction,
            seed=seed,
        )
    elif task_type == 'composition':
        dataset = ModularCompositionDataset(
            p=p,
            a1=np.random.randint(1, p),
            b1=np.random.randint(0, p),
            a2=np.random.randint(1, p),
            b2=np.random.randint(0, p),
            train_fraction=train_fraction,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Create model
    model = ModularMLP(
        input_dim=p,
        output_dim=p,
        hidden_dims=hidden_dims,
    )

    # Create dataloaders
    train_dataset = dataset.train()
    test_dataset = ModularArithmeticDataset(
        p=p, a=dataset.a if hasattr(dataset, 'a') else 1,
        b=dataset.b if hasattr(dataset, 'b') else 0,
        train_fraction=train_fraction, seed=seed
    ).test() if task_type == 'arithmetic' else \
        ModularCompositionDataset(
            p=p,
            a1=dataset.a1, b1=dataset.b1,
            a2=dataset.a2, b2=dataset.b2,
            train_fraction=train_fraction, seed=seed
        ).test()

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
    )

    return model, train_loader, test_loader


def evaluate_modular_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Evaluate model on modular task.

    Args:
        model: Neural network model
        dataloader: Test data loader
        device: Device to use

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
