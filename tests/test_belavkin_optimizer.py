"""
Unit tests for Belavkin optimizer.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from track1_optimizer.belavkin_optimizer import BelavkinOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestBelavkinOptimizer:
    """Test suite for BelavkinOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(
            model.parameters(),
            lr=1e-3,
            gamma=1e-4,
            beta=1e-2,
        )

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["gamma"] == 1e-4
        assert optimizer.param_groups[0]["beta"] == 1e-2

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        model = SimpleModel()

        with pytest.raises(ValueError):
            BelavkinOptimizer(model.parameters(), lr=-1.0)

        with pytest.raises(ValueError):
            BelavkinOptimizer(model.parameters(), gamma=-1.0)

        with pytest.raises(ValueError):
            BelavkinOptimizer(model.parameters(), beta=-1.0)

    def test_step(self):
        """Test that optimizer step updates parameters."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Create dummy input and loss
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        # Forward and backward
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Step
        optimizer.step()

        # Check that parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current), "Parameters should have changed"

    def test_zero_grad(self):
        """Test that zero_grad clears gradients."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(model.parameters())

        # Create gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None

        # Zero gradients
        optimizer.zero_grad()

        # Check gradients are zero or None
        for p in model.parameters():
            assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_adaptive_gamma(self):
        """Test adaptive gamma feature."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(
            model.parameters(),
            lr=1e-3,
            gamma=1e-4,
            adaptive_gamma=True,
        )

        # Run a few steps
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check that state was accumulated
        for p in model.parameters():
            state = optimizer.state[p]
            assert "grad_norm_history" in state
            assert len(state["grad_norm_history"]) > 0

    def test_weight_decay(self):
        """Test weight decay (L2 regularization)."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.1,
        )

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Parameters should have changed (no assertion on specific values)
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current)

    def test_state_statistics(self):
        """Test state statistics retrieval."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(model.parameters())

        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        stats = optimizer.get_state_statistics()

        assert "mean_grad_norm" in stats
        assert "max_grad_norm" in stats
        assert "total_steps" in stats
        assert stats["mean_grad_norm"] >= 0
        assert stats["max_grad_norm"] >= 0

    def test_multiple_param_groups(self):
        """Test optimizer with multiple parameter groups."""
        model = SimpleModel()
        optimizer = BelavkinOptimizer(
            [
                {"params": [model.linear.weight], "lr": 1e-3},
                {"params": [model.linear.bias], "lr": 1e-2},
            ]
        )

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
