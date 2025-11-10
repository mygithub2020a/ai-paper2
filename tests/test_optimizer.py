import torch
import torch.nn as nn
import pytest
from belavkin.optimizer import BelavkinOptimizer

def test_optimizer_updates_parameters():
    model = nn.Linear(10, 1)
    optimizer = BelavkinOptimizer(model.parameters())

    input_tensor = torch.randn(1, 10)
    target = torch.randn(1, 1)

    loss_fn = nn.MSELoss()
    loss = loss_fn(model(input_tensor), target)

    loss.backward()

    original_params = [p.clone() for p in model.parameters()]

    optimizer.step()

    for i, p in enumerate(model.parameters()):
        assert not torch.equal(p, original_params[i]), "Parameters were not updated"

def test_update_rule():
    model = nn.Linear(1, 1, bias=False)
    # Initialize weights to a known value
    with torch.no_grad():
        model.weight.fill_(1.0)

    optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, gamma=0.01, beta=0.0)

    input_tensor = torch.tensor([[2.0]])
    target = torch.tensor([[3.0]])

    loss_fn = nn.MSELoss()
    loss = loss_fn(model(input_tensor), target)
    loss.backward() # grad = 2 * (2*1 - 3) * 2 = -4

    # Manually calculate the expected change
    grad = model.weight.grad
    lr = 0.1
    gamma = 0.01

    damping = gamma * (grad ** 2)
    drift = lr * grad

    expected_change = -(damping + drift)

    original_weight = model.weight.clone()
    optimizer.step()

    assert torch.allclose(model.weight, original_weight + expected_change), "Update rule is not correct"

def test_adaptive_gamma():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, gamma=0.01, beta=0.0, adaptive_gamma=True, alpha=1.0)

    input_tensor = torch.tensor([[2.0]])
    target = torch.tensor([[3.0]])

    loss_fn = nn.MSELoss()
    loss = loss_fn(model(input_tensor), target)
    loss.backward()

    grad = model.weight.grad
    lr = 0.1
    gamma = 0.01
    alpha = 1.0

    grad_norm_sq = torch.sum(grad ** 2)
    expected_gamma = gamma / ((1 + grad_norm_sq) ** alpha)

    damping = expected_gamma * (grad ** 2)
    drift = lr * grad

    expected_change = -(damping + drift)

    original_weight = model.weight.clone()
    optimizer.step()

    assert torch.allclose(model.weight, original_weight + expected_change), "Adaptive gamma is not correct"

def test_adaptive_beta_not_implemented():
    model = nn.Linear(1, 1)
    optimizer = BelavkinOptimizer(model.parameters(), adaptive_beta=True)

    input_tensor = torch.randn(1, 1)
    target = torch.randn(1, 1)

    loss_fn = nn.MSELoss()
    loss = loss_fn(model(input_tensor), target)

    loss.backward()

    with pytest.raises(NotImplementedError):
        optimizer.step()
