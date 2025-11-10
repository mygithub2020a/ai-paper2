"""Tests for synthetic tasks."""

import pytest
import torch
from belavkin_ml.tasks.modular import (
    ModularArithmeticDataset,
    ModularCompositionDataset,
    create_modular_task,
)
from belavkin_ml.tasks.sparse_parity import (
    SparseParityDataset,
    create_sparse_parity_task,
)


def test_modular_arithmetic_dataset():
    """Test modular arithmetic dataset."""
    p = 13
    dataset = ModularArithmeticDataset(p=p, a=2, b=3, train_fraction=0.5, seed=42)

    # Check dataset size
    assert len(dataset) == p // 2  # Training set

    # Check data format
    x, y = dataset[0]
    assert x.shape == (p,)  # One-hot encoded
    assert y.shape == ()  # Scalar label
    assert 0 <= y < p


def test_modular_composition_dataset():
    """Test modular composition dataset."""
    p = 13
    dataset = ModularCompositionDataset(
        p=p, a1=2, b1=1, a2=3, b2=2, train_fraction=0.5, seed=42
    )

    assert len(dataset) == p // 2

    x, y = dataset[0]
    assert x.shape == (p,)
    assert y.shape == ()


def test_sparse_parity_dataset():
    """Test sparse parity dataset."""
    n_bits = 10
    k_sparse = 3
    n_samples = 100

    dataset = SparseParityDataset(
        n_bits=n_bits,
        k_sparse=k_sparse,
        n_samples=n_samples,
        train_fraction=0.8,
        seed=42,
    )

    # Check dataset
    assert len(dataset) == int(n_samples * 0.8)

    x, y = dataset[0]
    assert x.shape == (n_bits,)
    assert y.item() in [0, 1]

    # Check parity bits
    parity_bits = dataset.get_true_parity_bits()
    assert len(parity_bits) == k_sparse
    assert all(0 <= b < n_bits for b in parity_bits)


def test_create_modular_task():
    """Test modular task creation."""
    model, train_loader, test_loader = create_modular_task(
        task_type='arithmetic',
        p=13,
        hidden_dims=[32],
        train_fraction=0.5,
        batch_size=8,
        seed=42,
    )

    # Check model
    assert model is not None

    # Check dataloaders
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] <= 8  # Batch size
    assert batch_x.shape[1] == 13  # Input dimension


def test_create_sparse_parity_task():
    """Test sparse parity task creation."""
    model, train_loader, test_loader, dataset = create_sparse_parity_task(
        n_bits=10,
        k_sparse=3,
        n_samples=100,
        hidden_dims=[32],
        batch_size=8,
        seed=42,
    )

    # Check model
    assert model is not None

    # Check dataloaders
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[0] <= 8
    assert batch_x.shape[1] == 10

    # Check dataset
    assert len(dataset.get_true_parity_bits()) == 3


def test_dataset_train_test_split():
    """Test train/test split."""
    dataset = SparseParityDataset(
        n_bits=8,
        k_sparse=2,
        n_samples=100,
        train_fraction=0.7,
        seed=42,
    )

    train_len = len(dataset.train())
    test_len = len(dataset.test())

    assert train_len + test_len == 100
    assert train_len == 70


if __name__ == '__main__':
    pytest.main([__file__])
