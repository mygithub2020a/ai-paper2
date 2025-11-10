"""
Modular arithmetic datasets for testing optimizers.

Tasks:
- Addition: (a, b) -> (a + b) mod p
- Multiplication: (a, b) -> (a * b) mod p
- Inverse: a -> a^(-1) mod p
- Power: (a, k) -> a^k mod p
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class ModularArithmeticDataset(Dataset):
    """Base class for modular arithmetic datasets."""

    def __init__(
        self,
        n_samples: int,
        modulus: int,
        input_dim: int = 1,
        seed: Optional[int] = None,
        noise_level: float = 0.0,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            modulus: Prime modulus p
            input_dim: Dimension of input vectors (1 for scalar, >1 for vector)
            seed: Random seed for reproducibility
            noise_level: Label noise level (0.0 = no noise)
        """
        self.n_samples = n_samples
        self.modulus = modulus
        self.input_dim = input_dim
        self.noise_level = noise_level

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.inputs, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset. To be implemented by subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]

    def add_label_noise(self, labels: torch.Tensor) -> torch.Tensor:
        """Add label noise by randomly flipping labels."""
        if self.noise_level > 0:
            n_flip = int(self.n_samples * self.noise_level)
            flip_idx = np.random.choice(self.n_samples, n_flip, replace=False)
            # Replace with random labels
            labels[flip_idx] = torch.randint(0, self.modulus, (n_flip,))
        return labels


class ModularAddition(ModularArithmeticDataset):
    """Dataset for modular addition: (a, b) -> (a + b) mod p."""

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random inputs
        a = torch.randint(0, self.modulus, (self.n_samples, self.input_dim))
        b = torch.randint(0, self.modulus, (self.n_samples, self.input_dim))

        # Combine inputs: [a, b]
        inputs = torch.cat([a, b], dim=1).float()

        # Compute labels: (a + b) mod p
        labels = ((a + b) % self.modulus).float()

        # For multi-dimensional output, take the sum
        if self.input_dim > 1:
            labels = labels.sum(dim=1, keepdim=True)
            labels = (labels % self.modulus)

        labels = labels.squeeze()

        # Add label noise
        labels = self.add_label_noise(labels)

        return inputs, labels


class ModularMultiplication(ModularArithmeticDataset):
    """Dataset for modular multiplication: (a, b) -> (a * b) mod p."""

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, self.modulus, (self.n_samples, self.input_dim))
        b = torch.randint(0, self.modulus, (self.n_samples, self.input_dim))

        inputs = torch.cat([a, b], dim=1).float()

        # For simplicity, multiply element-wise and sum
        labels = ((a * b) % self.modulus).float()

        if self.input_dim > 1:
            labels = labels.sum(dim=1, keepdim=True)
            labels = (labels % self.modulus)

        labels = labels.squeeze()
        labels = self.add_label_noise(labels)

        return inputs, labels


class ModularInverse(ModularArithmeticDataset):
    """Dataset for modular inverse: a -> a^(-1) mod p."""

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate non-zero inputs (0 has no inverse)
        a = torch.randint(1, self.modulus, (self.n_samples, self.input_dim))

        inputs = a.float()

        # Compute modular inverse using extended Euclidean algorithm
        labels = torch.zeros(self.n_samples)
        for i in range(self.n_samples):
            if self.input_dim == 1:
                val = a[i, 0].item()
                labels[i] = pow(val, -1, self.modulus)
            else:
                # For vectors, take first element's inverse
                val = a[i, 0].item()
                labels[i] = pow(val, -1, self.modulus)

        labels = self.add_label_noise(labels)

        return inputs, labels


class ModularPower(ModularArithmeticDataset):
    """Dataset for modular power: (a, k) -> a^k mod p."""

    def __init__(
        self,
        n_samples: int,
        modulus: int,
        input_dim: int = 1,
        max_exponent: int = 100,
        seed: Optional[int] = None,
        noise_level: float = 0.0,
    ):
        self.max_exponent = max_exponent
        super().__init__(n_samples, modulus, input_dim, seed, noise_level)

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(1, self.modulus, (self.n_samples, self.input_dim))
        k = torch.randint(1, self.max_exponent, (self.n_samples, 1))

        inputs = torch.cat([a.float(), k.float()], dim=1)

        # Compute a^k mod p
        labels = torch.zeros(self.n_samples)
        for i in range(self.n_samples):
            if self.input_dim == 1:
                base = a[i, 0].item()
            else:
                # For vectors, use product
                base = (a[i].prod() % self.modulus).item()

            exp = k[i, 0].item()
            labels[i] = pow(int(base), int(exp), self.modulus)

        labels = self.add_label_noise(labels)

        return inputs, labels


def generate_modular_dataset(
    task: str,
    n_samples: int,
    modulus: int,
    input_dim: int = 1,
    batch_size: int = 32,
    seed: Optional[int] = None,
    noise_level: float = 0.0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate train and test dataloaders for a modular arithmetic task.

    Args:
        task: Task name ('add', 'mul', 'inv', 'pow')
        n_samples: Number of training samples
        modulus: Prime modulus
        input_dim: Input dimension
        batch_size: Batch size
        seed: Random seed
        noise_level: Label noise level
        **kwargs: Additional task-specific arguments

    Returns:
        (train_loader, test_loader)
    """
    task_map = {
        'add': ModularAddition,
        'mul': ModularMultiplication,
        'inv': ModularInverse,
        'pow': ModularPower,
    }

    if task not in task_map:
        raise ValueError(f"Unknown task: {task}. Choose from {list(task_map.keys())}")

    TaskClass = task_map[task]

    # Create train set
    train_dataset = TaskClass(
        n_samples=n_samples,
        modulus=modulus,
        input_dim=input_dim,
        seed=seed,
        noise_level=noise_level,
        **kwargs
    )

    # Create test set (10% of train size, different seed)
    test_seed = seed + 1000 if seed is not None else None
    test_dataset = TaskClass(
        n_samples=max(n_samples // 10, 100),
        modulus=modulus,
        input_dim=input_dim,
        seed=test_seed,
        noise_level=0.0,  # No noise in test set
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # Example usage
    train_loader, test_loader = generate_modular_dataset(
        task='add',
        n_samples=1000,
        modulus=97,
        input_dim=8,
        batch_size=32,
        seed=42
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Sample batch
    inputs, labels = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Sample input: {inputs[0]}")
    print(f"Sample label: {labels[0]}")
