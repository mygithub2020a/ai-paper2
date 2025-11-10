"""
Unit tests for synthetic tasks.
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from track1_optimizer.tasks.modular_arithmetic import (
    ModularArithmeticDataset,
    generate_modular_data,
    ModularMLP,
)
from track1_optimizer.tasks.modular_composition import (
    ModularCompositionDataset,
    generate_composition_data,
)
from track1_optimizer.tasks.sparse_parity import (
    SparseParityDataset,
    generate_parity_data,
)


class TestModularArithmetic:
    """Test modular arithmetic dataset."""

    def test_generate_data(self):
        """Test data generation."""
        X_train, y_train, X_test, y_test = generate_modular_data(
            prime=97, a=5, b=3, train_frac=0.5
        )

        assert len(X_train) + len(X_test) == 97
        assert torch.all((y_train >= 0) & (y_train < 97))
        assert torch.all((y_test >= 0) & (y_test < 97))

    def test_dataset(self):
        """Test ModularArithmeticDataset."""
        dataset = ModularArithmeticDataset(prime=97, train=True, train_frac=0.5)

        assert len(dataset) == 48  # 50% of 97 rounded down
        x, y = dataset[0]
        assert x.shape == (1,)
        assert isinstance(y.item(), int)

    def test_correctness(self):
        """Test that dataset computes correct modular arithmetic."""
        prime = 97
        a = 5
        b = 3

        dataset = ModularArithmeticDataset(prime=prime, a=a, b=b, train=True, train_frac=1.0)

        # Check a few samples
        for i in range(min(10, len(dataset))):
            x, y = dataset[i]
            x_val = int(x.item())
            expected = (a * x_val + b) % prime
            assert y.item() == expected, f"Mismatch at x={x_val}: got {y.item()}, expected {expected}"

    def test_model(self):
        """Test ModularMLP."""
        model = ModularMLP(input_dim=1, hidden_dim=64, output_dim=97)

        x = torch.randn(4, 1)
        output = model(x)

        assert output.shape == (4, 97)


class TestModularComposition:
    """Test modular composition dataset."""

    def test_generate_data(self):
        """Test composition data generation."""
        X_train, y_train, X_test, y_test = generate_composition_data(
            prime=97, a=5, b=3, c=7, d=2, train_frac=0.5
        )

        assert len(X_train) + len(X_test) == 97
        assert torch.all((y_train >= 0) & (y_train < 97))

    def test_dataset(self):
        """Test ModularCompositionDataset."""
        dataset = ModularCompositionDataset(prime=97, train=True, train_frac=0.5)

        assert len(dataset) > 0
        x, y = dataset[0]
        assert x.shape == (1,)

    def test_correctness(self):
        """Test that composition is computed correctly."""
        prime = 97
        a, b = 5, 3
        c, d = 7, 2

        dataset = ModularCompositionDataset(
            prime=prime, a=a, b=b, c=c, d=d, train=True, train_frac=1.0
        )

        # Check first sample
        x, y = dataset[0]
        x_val = int(x.item())

        # Compute manually: h(x) = f(g(x)) where g(x) = (cx + d) mod p, f(x) = (ax + b) mod p
        g_x = (c * x_val + d) % prime
        h_x = (a * g_x + b) % prime

        assert y.item() == h_x, f"Composition error at x={x_val}"


class TestSparseParity:
    """Test sparse parity dataset."""

    def test_generate_data(self):
        """Test parity data generation."""
        X_train, y_train, X_test, y_test, indices = generate_parity_data(
            n_bits=10, k_sparse=3, n_samples=100, train_frac=0.7
        )

        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(indices) == 3
        assert torch.all((y_train >= 0) & (y_train <= 1))

    def test_dataset(self):
        """Test SparseParityDataset."""
        dataset = SparseParityDataset(
            n_bits=10, k_sparse=3, n_samples=100, train=True, train_frac=0.7
        )

        assert len(dataset) == 70
        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.item() in [0, 1]

    def test_parity_correctness(self):
        """Test that parity is computed correctly."""
        dataset = SparseParityDataset(
            n_bits=10,
            k_sparse=3,
            n_samples=100,
            train=True,
            train_frac=1.0,
            parity_indices=[2, 5, 7],
        )

        # Check a few samples
        for i in range(min(10, len(dataset))):
            x, y = dataset[i]

            # Manually compute parity
            manual_parity = 0
            for idx in dataset.parity_indices:
                manual_parity ^= int(x[idx].item())

            assert y.item() == manual_parity, f"Parity mismatch at sample {i}"

    def test_fixed_indices(self):
        """Test that fixed parity indices work."""
        indices = [1, 3, 5]
        dataset = SparseParityDataset(
            n_bits=10,
            k_sparse=3,
            n_samples=100,
            train=True,
            parity_indices=indices,
        )

        assert dataset.parity_indices == indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
