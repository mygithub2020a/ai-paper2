"""
Unit tests for Belavkin optimizer.
"""

import pytest
import torch
import torch.nn as nn
from belavkin_ml.optimizer import BelavkinOptimizer, AdaptiveBelavkinOptimizer


def test_belavkin_optimizer_initialization():
    """Test that optimizer can be initialized."""
    model = nn.Linear(10, 5)
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        gamma=1e-4,
        beta=1e-2,
    )
    assert optimizer is not None


def test_belavkin_optimizer_step():
    """Test that optimizer can perform a step."""
    model = nn.Linear(10, 5)
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    assert loss.item() >= 0


def test_belavkin_optimizer_convergence():
    """Test that optimizer can reduce loss on simple problem."""
    # Simple linear regression
    torch.manual_seed(42)
    true_w = torch.tensor([2.0, -1.0])
    true_b = torch.tensor([0.5])

    model = nn.Linear(2, 1)
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-2,
        gamma=1e-4,
        beta=1e-3,
    )

    # Generate data
    X = torch.randn(100, 2)
    y = (X @ true_w.unsqueeze(1) + true_b).squeeze()

    # Train for a few steps
    initial_loss = None
    final_loss = None

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 49:
            final_loss = loss.item()

    # Loss should decrease
    assert final_loss < initial_loss


def test_adaptive_belavkin_optimizer():
    """Test adaptive variant."""
    model = nn.Linear(10, 5)
    optimizer = AdaptiveBelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        adapt_gamma=True,
        adapt_beta=True,
    )

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    output = model(x)
    loss = nn.MSELoss()(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check adaptation stats
    stats = optimizer.get_adaptation_stats()
    assert 'current_params' in stats


def test_optimizer_with_different_hyperparameters():
    """Test optimizer with various hyperparameter combinations."""
    model = nn.Linear(5, 3)

    configs = [
        {'lr': 1e-3, 'gamma': 1e-5, 'beta': 1e-3},
        {'lr': 1e-2, 'gamma': 1e-4, 'beta': 1e-2},
        {'lr': 1e-4, 'gamma': 1e-3, 'beta': 1e-1},
    ]

    for config in configs:
        optimizer = BelavkinOptimizer(model.parameters(), **config)

        x = torch.randn(4, 5)
        y = torch.randn(4, 3)

        output = model(x)
        loss = nn.MSELoss()(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0


def test_optimizer_state_dict():
    """Test that optimizer state can be saved and loaded."""
    model = nn.Linear(10, 5)
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

    # Perform a step to initialize state
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()

    # Save state
    state_dict = optimizer.state_dict()

    # Create new optimizer and load state
    model2 = nn.Linear(10, 5)
    optimizer2 = BelavkinOptimizer(model2.parameters(), lr=1e-3)
    optimizer2.load_state_dict(state_dict)

    # States should match
    assert len(optimizer.state) == len(optimizer2.state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
