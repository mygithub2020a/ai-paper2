"""Test basic optimization functionality of BelOpt."""

import pytest
import torch
import torch.nn as nn
from belavkin.belopt import BelOpt


def test_simple_quadratic():
    """Test optimization on a simple quadratic function."""
    # Minimize f(x) = sum(x^2)
    x = torch.randn(10, requires_grad=True) * 10  # Start far from optimum
    optimizer = BelOpt([x], lr=0.1, deterministic=True)

    initial_norm = torch.norm(x).item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        optimizer.step()

    final_norm = torch.norm(x).item()

    # Should move towards zero
    assert final_norm < initial_norm
    assert final_norm < 1.0  # Should get reasonably close


def test_linear_regression():
    """Test on simple linear regression problem."""
    torch.manual_seed(42)

    # Generate data: y = 3x + 2 + noise
    n_samples = 100
    x_data = torch.randn(n_samples, 1)
    y_data = 3 * x_data + 2 + torch.randn(n_samples, 1) * 0.1

    # Model
    model = nn.Linear(1, 1)
    optimizer = BelOpt(model.parameters(), lr=0.01, deterministic=True)

    # Train
    for _ in range(200):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = ((pred - y_data) ** 2).mean()
        loss.backward()
        optimizer.step()

    # Check learned parameters are close to true values
    with torch.no_grad():
        weight = model.weight.item()
        bias = model.bias.item()

    assert abs(weight - 3.0) < 0.5
    assert abs(bias - 2.0) < 0.5


def test_parameter_update():
    """Test that parameters actually update."""
    param = torch.randn(10, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param_before = param.clone()
    param.grad = torch.randn(10)
    optimizer.step()

    # Parameter should change
    assert not torch.allclose(param, param_before)


def test_zero_lr():
    """Test that zero learning rate doesn't update parameters."""
    param = torch.randn(10, requires_grad=True)
    optimizer = BelOpt([param], lr=0.0, deterministic=True)

    param_before = param.clone()
    param.grad = torch.randn(10)
    optimizer.step()

    # Parameter should not change (assuming no weight decay)
    assert torch.allclose(param, param_before)


def test_weight_decay():
    """Test weight decay functionality."""
    param = torch.ones(10, requires_grad=True) * 10.0
    optimizer = BelOpt(
        [param],
        lr=0.1,
        weight_decay=0.1,
        decoupled_weight_decay=True,
        deterministic=True
    )

    param.grad = torch.zeros(10)  # Zero gradient
    param_before_norm = torch.norm(param).item()

    optimizer.step()

    param_after_norm = torch.norm(param).item()

    # Weight decay should reduce parameter magnitude even with zero gradient
    assert param_after_norm < param_before_norm


def test_gradient_clipping():
    """Test gradient clipping."""
    param = torch.zeros(10, requires_grad=True)
    optimizer = BelOpt([param], lr=1.0, grad_clip=1.0, deterministic=True)

    # Large gradient
    param.grad = torch.ones(10) * 100.0

    optimizer.step()

    # Update should be bounded due to gradient clipping
    param_change = torch.norm(param).item()
    # With grad_clip=1.0, the gradient norm is clipped to 1.0
    # So the change should be roughly lr * clipped_grad_norm = 1.0 * 1.0 = 1.0
    assert param_change < 5.0  # Should be small due to clipping


def test_multiple_param_groups():
    """Test with multiple parameter groups."""
    param1 = torch.randn(10, requires_grad=True)
    param2 = torch.randn(20, requires_grad=True)

    optimizer = BelOpt([
        {'params': [param1], 'lr': 0.1},
        {'params': [param2], 'lr': 0.01},
    ])

    param1.grad = torch.randn(10)
    param2.grad = torch.randn(20)

    param1_before = param1.clone()
    param2_before = param2.clone()

    optimizer.step()

    # Both should update
    assert not torch.allclose(param1, param1_before)
    assert not torch.allclose(param2, param2_before)


def test_closure():
    """Test optimization with closure."""
    x = torch.randn(10, requires_grad=True)
    optimizer = BelOpt([x], lr=0.1, deterministic=True)

    def closure():
        optimizer.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    initial_loss = closure().item()
    optimizer.step(closure)

    # Check that loss decreased
    current_loss = (x ** 2).sum().item()
    assert current_loss < initial_loss


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
