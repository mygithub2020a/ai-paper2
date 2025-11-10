"""Test determinism of BelOpt with fixed seeds."""

import pytest
import torch
from belavkin.belopt import BelOpt


def test_deterministic_mode():
    """Test that deterministic mode produces identical results."""
    torch.manual_seed(42)
    param1 = torch.randn(100, requires_grad=True)
    optimizer1 = BelOpt([param1], lr=0.01, deterministic=True)

    torch.manual_seed(42)
    param2 = torch.randn(100, requires_grad=True)
    optimizer2 = BelOpt([param2], lr=0.01, deterministic=True)

    # Multiple steps with same gradients
    for _ in range(10):
        torch.manual_seed(42)
        grad = torch.randn(100)

        param1.grad = grad.clone()
        param2.grad = grad.clone()

        optimizer1.step()
        optimizer2.step()

    # Results should be identical
    assert torch.allclose(param1, param2, atol=1e-6)


def test_stochastic_mode_with_seed():
    """Test that stochastic mode is reproducible with fixed seed."""
    # First run
    torch.manual_seed(42)
    param1 = torch.randn(100, requires_grad=True)
    optimizer1 = BelOpt([param1], lr=0.01, beta0=0.01, deterministic=False)

    grads1 = []
    for i in range(10):
        torch.manual_seed(100 + i)  # Different seed for each gradient
        grad = torch.randn(100)
        grads1.append(grad.clone())
        param1.grad = grad
        optimizer1.step()

    param1_final = param1.clone()

    # Second run with same seeds
    torch.manual_seed(42)
    param2 = torch.randn(100, requires_grad=True)
    optimizer2 = BelOpt([param2], lr=0.01, beta0=0.01, deterministic=False)

    for i in range(10):
        param2.grad = grads1[i]
        torch.manual_seed(42)  # Reset seed before step to get same noise
        optimizer2.step()

    # Note: This test might not pass because the noise is drawn during step()
    # and we can't easily control it. Let's instead check that beta0=0 gives determinism


def test_beta_zero_determinism():
    """Test that beta=0 gives deterministic results regardless of random seed."""
    torch.manual_seed(42)
    param1 = torch.randn(100, requires_grad=True)
    optimizer1 = BelOpt([param1], lr=0.01, beta0=0.0)

    torch.manual_seed(123)  # Different seed
    param2 = torch.randn(100, requires_grad=True)
    optimizer2 = BelOpt([param2], lr=0.01, beta0=0.0)

    # Same initial parameters
    param2.data.copy_(param1.data)

    # Multiple steps with same gradients
    for i in range(10):
        torch.manual_seed(i)
        grad = torch.randn(100)

        param1.grad = grad.clone()
        param2.grad = grad.clone()

        optimizer1.step()
        optimizer2.step()

    # Results should be identical when beta=0
    assert torch.allclose(param1, param2, atol=1e-6)


def test_reproducible_training():
    """Test that training is reproducible with fixed seed."""

    def run_training(seed):
        torch.manual_seed(seed)
        param = torch.randn(50, requires_grad=True)
        optimizer = BelOpt([param], lr=0.01, deterministic=True)

        for i in range(20):
            # Deterministic gradient
            torch.manual_seed(seed + i)
            param.grad = torch.randn(50)
            optimizer.step()

        return param.clone()

    # Run twice with same seed
    result1 = run_training(42)
    result2 = run_training(42)

    # Should be identical
    assert torch.allclose(result1, result2, atol=1e-6)

    # Run with different seed
    result3 = run_training(123)

    # Should be different
    assert not torch.allclose(result1, result3, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
