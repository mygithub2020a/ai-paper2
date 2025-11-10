"""
Unit tests for synthetic datasets.
"""

import pytest
import torch
from belavkin_ml.datasets.synthetic import (
    ModularArithmeticDataset,
    ModularCompositionDataset,
    SparseParityDataset,
    create_dataloaders,
)


def test_modular_arithmetic_creation():
    """Test that modular arithmetic dataset can be created."""
    dataset = ModularArithmeticDataset(p=13, operation='addition')
    assert len(dataset) > 0
    assert dataset.p == 13


def test_modular_arithmetic_operations():
    """Test different modular arithmetic operations."""
    operations = ['addition', 'multiplication', 'division']

    for op in operations:
        dataset = ModularArithmeticDataset(p=13, operation=op)
        x, y = dataset[0]

        assert x.shape[0] == 2  # Two inputs
        assert 0 <= y < 13  # Output in valid range


def test_modular_arithmetic_train_test_split():
    """Test train/test split."""
    dataset = ModularArithmeticDataset(p=13, operation='addition', train_fraction=0.5)

    dataset.set_train()
    train_len = len(dataset)

    dataset.set_test()
    test_len = len(dataset)

    # Should have roughly equal split
    total_len = train_len + test_len
    assert abs(train_len - test_len) < 0.2 * total_len


def test_modular_composition():
    """Test modular composition dataset."""
    dataset = ModularCompositionDataset(p=13, f_type='linear', g_type='linear')

    x, y = dataset[0]
    assert x.shape[0] == 1  # Single input
    assert 0 <= y < 13


def test_sparse_parity():
    """Test sparse parity dataset."""
    dataset = SparseParityDataset(n_bits=10, k_sparse=3, n_examples=100)

    x, y = dataset[0]
    assert x.shape[0] == 10  # n_bits
    assert y in [0, 1]  # Binary output

    info = dataset.get_info()
    assert len(info['relevant_bits']) == 3


def test_sparse_parity_correctness():
    """Test that sparse parity function is correct."""
    dataset = SparseParityDataset(n_bits=10, k_sparse=3, n_examples=100, seed=42)

    # Check a few examples
    for i in range(10):
        x, y = dataset[i]

        # Compute expected parity
        relevant_bits = dataset.relevant_bits
        relevant_values = x[relevant_bits]
        expected_parity = int(torch.sum(relevant_values).item()) % 2

        assert y == expected_parity


def test_create_dataloaders():
    """Test dataloader creation."""
    dataset = ModularArithmeticDataset(p=13, operation='addition')

    train_loader, test_loader = create_dataloaders(dataset, batch_size=32)

    assert len(train_loader) > 0
    assert len(test_loader) > 0

    # Check batch shape
    for x, y in train_loader:
        assert x.shape[1] == 2  # Input dimension
        assert y.shape[0] <= 32  # Batch size
        break


def test_dataset_reproducibility():
    """Test that datasets are reproducible with same seed."""
    dataset1 = ModularArithmeticDataset(p=13, operation='addition', seed=42)
    dataset2 = ModularArithmeticDataset(p=13, operation='addition', seed=42)

    dataset1.set_train()
    dataset2.set_train()

    # Should produce same examples
    for i in range(min(10, len(dataset1))):
        x1, y1 = dataset1[i]
        x2, y2 = dataset2[i]

        assert torch.equal(x1, x2)
        assert y1 == y2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
