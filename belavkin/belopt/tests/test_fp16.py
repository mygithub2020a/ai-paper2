"""Test FP16/mixed precision compatibility of BelOpt."""

import pytest
import torch
from belavkin.belopt import BelOpt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_cuda():
    """Test with FP16 on CUDA."""
    device = torch.device('cuda')
    param = torch.randn(100, dtype=torch.float16, device=device, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    param.grad = torch.randn(100, dtype=torch.float16, device=device)
    optimizer.step()

    assert param.dtype == torch.float16
    assert not torch.isnan(param).any()
    assert not torch.isinf(param).any()


def test_fp16_cpu():
    """Test with FP16 on CPU (should work but may have precision issues)."""
    param = torch.randn(100, dtype=torch.float16, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01, eps=1e-4)  # Larger eps for FP16

    param.grad = torch.randn(100, dtype=torch.float16)

    # Should not crash
    optimizer.step()

    assert param.dtype == torch.float16


def test_no_nan_or_inf():
    """Test that optimizer doesn't produce NaN or Inf values."""
    param = torch.randn(100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    for _ in range(100):
        param.grad = torch.randn(100)
        optimizer.step()

    assert not torch.isnan(param).any()
    assert not torch.isinf(param).any()


def test_large_gradients():
    """Test behavior with very large gradients."""
    param = torch.randn(100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01, grad_clip=1.0)

    # Very large gradient
    param.grad = torch.randn(100) * 1e6

    optimizer.step()

    # Should still be finite due to gradient clipping
    assert not torch.isnan(param).any()
    assert not torch.isinf(param).any()


def test_small_gradients():
    """Test behavior with very small gradients."""
    param = torch.randn(100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    # Very small gradient
    param.grad = torch.randn(100) * 1e-10

    optimizer.step()

    assert not torch.isnan(param).any()
    assert not torch.isinf(param).any()


def test_mixed_precision_simulation():
    """Simulate mixed precision training (FP32 weights, FP16 gradients)."""
    # In real mixed precision, weights are FP32 but gradients might be FP16
    param = torch.randn(100, dtype=torch.float32, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01)

    # Simulate FP16 gradient by converting to FP16 and back
    grad_fp16 = torch.randn(100, dtype=torch.float16)
    param.grad = grad_fp16.to(torch.float32)

    optimizer.step()

    assert param.dtype == torch.float32
    assert not torch.isnan(param).any()


def test_numerical_stability():
    """Test numerical stability with edge case values."""
    param = torch.randn(100, requires_grad=True)
    optimizer = BelOpt([param], lr=0.01, eps=1e-8)

    # Mix of large and small values
    param.grad = torch.cat([
        torch.randn(25) * 1e-8,
        torch.randn(25) * 1.0,
        torch.randn(25) * 1e3,
        torch.randn(25) * 1e-3,
    ])

    optimizer.step()

    assert not torch.isnan(param).any()
    assert not torch.isinf(param).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
