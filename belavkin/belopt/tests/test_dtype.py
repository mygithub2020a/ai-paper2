"""Test that BelOpt handles different dtypes correctly."""

import pytest
import torch
from belavkin.belopt import BelOpt


def test_float32():
    """Test with float32 parameters."""
    param = torch.randn(10, dtype=torch.float32, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(10, dtype=torch.float32)
    optimizer.step()

    assert param.dtype == torch.float32


def test_float64():
    """Test with float64 parameters."""
    param = torch.randn(10, dtype=torch.float64, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(10, dtype=torch.float64)
    optimizer.step()

    assert param.dtype == torch.float64


def test_mixed_dtype():
    """Test with mixed dtype parameters."""
    param1 = torch.randn(10, dtype=torch.float32, requires_grad=True)
    param2 = torch.randn(10, dtype=torch.float64, requires_grad=True)

    optimizer = BelOpt([param1, param2], lr=0.01)

    param1.grad = torch.randn(10, dtype=torch.float32)
    param2.grad = torch.randn(10, dtype=torch.float64)

    optimizer.step()

    assert param1.dtype == torch.float32
    assert param2.dtype == torch.float64


def test_dtype_consistency():
    """Test that operations maintain dtype consistency."""
    for dtype in [torch.float32, torch.float64]:
        param = torch.randn(100, dtype=dtype, requires_grad=True)
        optimizer = BelOpt([param], lr=0.01, adaptive_gamma=True)

        # Multiple steps
        for _ in range(5):
            param.grad = torch.randn(100, dtype=dtype)
            optimizer.step()

        # Check dtype is preserved
        assert param.dtype == dtype

        # Check state variables have correct dtype
        state = optimizer.state[param]
        if 'v' in state:
            assert state['v'].dtype == dtype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
