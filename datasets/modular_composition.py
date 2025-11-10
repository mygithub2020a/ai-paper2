"""
Modular Composition Dataset Generator

Generates synthetic datasets for learning compositions of modular functions:
- f(g(x)) mod p where f and g are modular operations
- Nested compositions: f(g(h(x))) mod p
- Mixed operations: (a * b + c) mod p

These tasks test optimizer's ability to learn hierarchical algebraic structures.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Callable, List


class ModularCompositionDataset(Dataset):
    """
    Dataset for composed modular operations.

    Args:
        composition_type (str): Type of composition
            - 'two_layer': f(g(x)) mod p
            - 'three_layer': f(g(h(x))) mod p
            - 'mixed': (a * b + c) mod p
        modulus (int): The modulus p
        num_samples (int): Number of samples
        seed (Optional[int]): Random seed
    """

    def __init__(
        self,
        composition_type: str = 'two_layer',
        modulus: int = 97,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ):
        valid_types = ['two_layer', 'three_layer', 'mixed', 'polynomial']
        if composition_type not in valid_types:
            raise ValueError(f"Invalid composition_type. Must be one of {valid_types}")

        self.composition_type = composition_type
        self.modulus = modulus
        self.num_samples = num_samples

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.data, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate compositional modular data."""
        if self.composition_type == 'two_layer':
            return self._generate_two_layer()
        elif self.composition_type == 'three_layer':
            return self._generate_three_layer()
        elif self.composition_type == 'mixed':
            return self._generate_mixed()
        elif self.composition_type == 'polynomial':
            return self._generate_polynomial()

    def _generate_two_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate f(g(x)) = (2 * (x + 1)) mod p
        Input: x
        Output: (2 * (x + 1)) mod p
        """
        x = torch.randint(0, self.modulus, (self.num_samples,))

        # g(x) = (x + 1) mod p
        g_x = (x + 1) % self.modulus

        # f(y) = (2 * y) mod p
        labels = (2 * g_x) % self.modulus

        data = x.unsqueeze(1).float()
        return data, labels

    def _generate_three_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate f(g(h(x))) = (3 * ((x * 2) + 1)) mod p
        Input: x
        Output: (3 * ((x * 2) + 1)) mod p
        """
        x = torch.randint(0, self.modulus, (self.num_samples,))

        # h(x) = (x * 2) mod p
        h_x = (x * 2) % self.modulus

        # g(y) = (y + 1) mod p
        g_h_x = (h_x + 1) % self.modulus

        # f(z) = (3 * z) mod p
        labels = (3 * g_h_x) % self.modulus

        data = x.unsqueeze(1).float()
        return data, labels

    def _generate_mixed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mixed operation: (a * b + c) mod p
        Input: (a, b, c)
        Output: (a * b + c) mod p
        """
        a = torch.randint(0, self.modulus, (self.num_samples,))
        b = torch.randint(0, self.modulus, (self.num_samples,))
        c = torch.randint(0, self.modulus, (self.num_samples,))

        labels = ((a * b) + c) % self.modulus

        data = torch.stack([a, b, c], dim=1).float()
        return data, labels

    def _generate_polynomial(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate polynomial: (x^2 + 3*x + 5) mod p
        Input: x
        Output: (x^2 + 3*x + 5) mod p
        """
        x = torch.randint(0, self.modulus, (self.num_samples,))

        labels = (x.pow(2) + 3 * x + 5) % self.modulus

        data = x.unsqueeze(1).float()
        return data, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def generate_composition_data(
    composition_type: str = 'two_layer',
    modulus: int = 97,
    num_samples: int = 10000,
    train_split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[ModularCompositionDataset, ModularCompositionDataset]:
    """
    Generate train and test datasets for modular composition.

    Args:
        composition_type: Type of composition to generate
        modulus: The modulus for modular arithmetic
        num_samples: Total number of samples
        train_split: Fraction of data for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_samples = int(num_samples * train_split)
    test_samples = num_samples - train_samples

    train_ds = ModularCompositionDataset(
        composition_type=composition_type,
        modulus=modulus,
        num_samples=train_samples,
        seed=seed,
    )

    test_ds = ModularCompositionDataset(
        composition_type=composition_type,
        modulus=modulus,
        num_samples=test_samples,
        seed=seed + 1 if seed is not None else None,
    )

    return train_ds, test_ds


class AdvancedModularComposition(Dataset):
    """
    Advanced compositional tasks with custom function composition.

    This allows for more complex testing scenarios.
    """

    def __init__(
        self,
        functions: List[Callable[[torch.Tensor, int], torch.Tensor]],
        modulus: int = 97,
        num_inputs: int = 1,
        num_samples: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            functions: List of functions to compose (applied right to left)
            modulus: The modulus
            num_inputs: Number of input variables
            num_samples: Number of samples
            seed: Random seed
        """
        self.functions = functions
        self.modulus = modulus
        self.num_inputs = num_inputs
        self.num_samples = num_samples

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.data, self.labels = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data by composing functions."""
        # Generate random inputs
        if self.num_inputs == 1:
            inputs = torch.randint(0, self.modulus, (self.num_samples,))
            data = inputs.unsqueeze(1).float()
        else:
            inputs = torch.randint(0, self.modulus, (self.num_samples, self.num_inputs))
            data = inputs.float()

        # Apply functions in composition (right to left)
        result = inputs if self.num_inputs == 1 else inputs[:, 0]
        for func in reversed(self.functions):
            result = func(result, self.modulus)

        labels = result % self.modulus
        return data, labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
