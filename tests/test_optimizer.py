"""Tests for Belavkin optimizer."""

import pytest
import torch
import torch.nn as nn
from belavkin_ml.optimizers.belavkin import BelavkinOptimizer, BelavkinSGD, BelavkinAdam


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_belavkin_optimizer_initialization():
    """Test optimizer initialization."""
    model = SimpleModel()
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

    assert optimizer.defaults['lr'] == 1e-3
    assert optimizer.defaults['gamma'] == 1e-4
    assert optimizer.defaults['beta'] == 1e-2


def test_belavkin_optimizer_step():
    """Test optimizer step."""
    model = SimpleModel()
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    # Forward pass
    output = model(x)
    loss = ((output - y) ** 2).mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None

    # Take optimizer step
    optimizer.step()

    # Check parameters updated
    # (Can't check exact values due to stochasticity)


def test_belavkin_sgd_variant():
    """Test BelavkinSGD variant."""
    model = SimpleModel()
    optimizer = BelavkinSGD(model.parameters(), lr=1e-2)

    assert optimizer.defaults['lr'] == 1e-2
    assert optimizer.defaults['adaptive_gamma'] is True


def test_belavkin_adam_variant():
    """Test BelavkinAdam variant."""
    model = SimpleModel()
    optimizer = BelavkinAdam(model.parameters(), lr=1e-3)

    assert optimizer.defaults['amsgrad'] is True
    assert optimizer.defaults['adaptive_beta'] is True


def test_optimizer_state_accumulation():
    """Test that optimizer maintains state."""
    model = SimpleModel()
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    # First step
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Check state exists
    for param in model.parameters():
        state = optimizer.state[param]
        assert 'step' in state
        assert 'grad_sq_sum' in state
        assert state['step'] == 1


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    model = SimpleModel()
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        grad_clip=1.0,
    )

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean() * 1000  # Large loss
    loss.backward()
    optimizer.step()

    # Should not raise error despite large gradients


def test_weight_decay():
    """Test weight decay."""
    model = SimpleModel()
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()


def test_invalid_hyperparameters():
    """Test that invalid hyperparameters raise errors."""
    model = SimpleModel()

    with pytest.raises(ValueError):
        BelavkinOptimizer(model.parameters(), lr=-1.0)

    with pytest.raises(ValueError):
        BelavkinOptimizer(model.parameters(), gamma=-1.0)

    with pytest.raises(ValueError):
        BelavkinOptimizer(model.parameters(), beta=-1.0)


if __name__ == '__main__':
    pytest.main([__file__])
