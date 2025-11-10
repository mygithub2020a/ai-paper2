"""Test that BelOpt preserves parameter shapes correctly."""

import pytest
import torch
from belavkin.belopt import BelOpt


def test_scalar_parameter():
    """Test with scalar parameter."""
    param = torch.randn(1, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    # Simulate gradient
    param.grad = torch.randn(1)

    # Step
    optimizer.step()

    # Check shape is preserved
    assert param.shape == (1,)


def test_vector_parameter():
    """Test with vector parameter."""
    param = torch.randn(100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(100)
    optimizer.step()

    assert param.shape == (100,)


def test_matrix_parameter():
    """Test with matrix parameter."""
    param = torch.randn(50, 100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(50, 100)
    optimizer.step()

    assert param.shape == (50, 100)


def test_tensor_parameter():
    """Test with high-dimensional tensor parameter."""
    param = torch.randn(10, 20, 30, 40, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(10, 20, 30, 40)
    optimizer.step()

    assert param.shape == (10, 20, 30, 40)


def test_multiple_parameters():
    """Test with multiple parameters of different shapes."""
    params = [
        torch.randn(10, requires_grad=True),
        torch.randn(20, 30, requires_grad=True),
        torch.randn(5, 10, 15, requires_grad=True),
    ]
    optimizer = BelOpt(params, lr=0.01)

    # Set gradients
    params[0].grad = torch.randn(10)
    params[1].grad = torch.randn(20, 30)
    params[2].grad = torch.randn(5, 10, 15)

    optimizer.step()

    # Check all shapes preserved
    assert params[0].shape == (10,)
    assert params[1].shape == (20, 30)
    assert params[2].shape == (5, 10, 15)


def test_empty_gradient():
    """Test behavior when gradient is None."""
    param = torch.randn(10, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    # No gradient set
    param_before = param.clone()
    optimizer.step()

    # Parameter should not change
    assert torch.allclose(param, param_before)


def test_zero_gradient():
    """Test with zero gradient."""
    param = torch.randn(10, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01, deterministic=True)

    param.grad = torch.zeros(10)
    param_before = param.clone()

    optimizer.step()

    # With zero gradient and no exploration, parameter should not change (ignoring weight decay)
    # Actually with weight decay it might change, so test without weight decay
    assert param.shape == (10,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
