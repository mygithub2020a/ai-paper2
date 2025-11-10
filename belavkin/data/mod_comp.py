"""
Modular composition dataset: learn y = f(g(x)) mod p

where f and g are random low-degree polynomials or affine maps.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable


class PolynomialFunction:
    """Random polynomial function modulo p."""

    def __init__(self, degree: int, modulus: int, input_dim: int, seed: Optional[int] = None):
        """
        Args:
            degree: Polynomial degree
            modulus: Prime modulus
            input_dim: Input dimension
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.degree = degree
        self.modulus = modulus
        self.input_dim = input_dim

        # Generate random coefficients
        # For simplicity, use linear combination of monomials
        self.coeffs = np.random.randint(0, modulus, size=(degree + 1, input_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at x.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size,)
        """
        result = np.zeros(x.shape[0], dtype=np.int64)

        for d in range(self.degree + 1):
            # Compute x^d term
            term = np.power(x, d) % self.modulus
            # Weight by coefficients and sum over input dimensions
            weighted = (term * self.coeffs[d]) % self.modulus
            result = (result + weighted.sum(axis=1)) % self.modulus

        return result


class AffineFunction:
    """Random affine function: f(x) = Ax + b mod p."""

    def __init__(self, modulus: int, input_dim: int, output_dim: int, seed: Optional[int] = None):
        """
        Args:
            modulus: Prime modulus
            input_dim: Input dimension
            output_dim: Output dimension
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.modulus = modulus
        self.A = np.random.randint(0, modulus, size=(output_dim, input_dim))
        self.b = np.random.randint(0, modulus, size=(output_dim,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate affine function at x.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size, output_dim) or (batch_size,) if output_dim=1
        """
        result = (x @ self.A.T + self.b) % self.modulus

        if result.shape[1] == 1:
            result = result.squeeze(1)

        return result


class ModularComposition(Dataset):
    """Dataset for modular composition: y = f(g(x)) mod p."""

    def __init__(
        self,
        n_samples: int,
        modulus: int,
        input_dim: int = 1,
        composition_depth: int = 2,
        function_type: str = 'polynomial',
        poly_degree: int = 2,
        seed: Optional[int] = None,
        noise_level: float = 0.0,
    ):
        """
        Args:
            n_samples: Number of samples
            modulus: Prime modulus
            input_dim: Input dimension
            composition_depth: Depth of composition (2 = f(g(x)), 3 = f(g(h(x))), etc.)
            function_type: 'polynomial' or 'affine'
            poly_degree: Degree of polynomial (if function_type='polynomial')
            seed: Random seed
            noise_level: Label noise level
        """
        self.n_samples = n_samples
        self.modulus = modulus
        self.input_dim = input_dim
        self.composition_depth = composition_depth
        self.function_type = function_type
        self.poly_degree = poly_degree
        self.noise_level = noise_level

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create composition chain
        self.functions = []
        for i in range(composition_depth):
            func_seed = seed + i if seed is not None else None
            if function_type == 'polynomial':
                func = PolynomialFunction(poly_degree, modulus, input_dim, func_seed)
            elif function_type == 'affine':
                func = AffineFunction(modulus, input_dim, input_dim, func_seed)
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            self.functions.append(func)

        self.inputs, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset by composing functions."""
        # Generate random inputs
        inputs = np.random.randint(0, self.modulus, size=(self.n_samples, self.input_dim))

        # Apply composition
        x = inputs.copy()
        for func in self.functions:
            x = func(x)
            # Ensure x is 2D for next iteration
            if x.ndim == 1:
                x = x.reshape(-1, 1)
                # Expand back to input_dim by repeating
                if x.shape[1] < self.input_dim:
                    x = np.tile(x, (1, self.input_dim))

        # Final output
        labels = x if x.ndim == 1 else x[:, 0]

        # Add label noise
        if self.noise_level > 0:
            n_flip = int(self.n_samples * self.noise_level)
            flip_idx = np.random.choice(self.n_samples, n_flip, replace=False)
            labels[flip_idx] = np.random.randint(0, self.modulus, size=n_flip)

        # Convert to tensors
        inputs_tensor = torch.from_numpy(inputs).float()
        labels_tensor = torch.from_numpy(labels).float()

        return inputs_tensor, labels_tensor

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


def generate_composition_dataset(
    n_samples: int,
    modulus: int,
    input_dim: int = 1,
    composition_depth: int = 2,
    batch_size: int = 32,
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate train and test dataloaders for modular composition.

    Args:
        n_samples: Number of training samples
        modulus: Prime modulus
        input_dim: Input dimension
        composition_depth: Depth of composition
        batch_size: Batch size
        seed: Random seed
        **kwargs: Additional arguments for ModularComposition

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = ModularComposition(
        n_samples=n_samples,
        modulus=modulus,
        input_dim=input_dim,
        composition_depth=composition_depth,
        seed=seed,
        **kwargs
    )

    test_seed = seed + 1000 if seed is not None else None
    test_dataset = ModularComposition(
        n_samples=max(n_samples // 10, 100),
        modulus=modulus,
        input_dim=input_dim,
        composition_depth=composition_depth,
        seed=test_seed,
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
    train_loader, test_loader = generate_composition_dataset(
        n_samples=1000,
        modulus=97,
        input_dim=8,
        composition_depth=3,
        function_type='polynomial',
        poly_degree=2,
        batch_size=32,
        seed=42
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    inputs, labels = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
