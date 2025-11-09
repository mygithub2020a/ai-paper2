import torch
from belavkin_optimizer.optimizer import BelavkinOptimizer

def test_optimizer_on_quadratic():
    # Define a simple quadratic function: f(x) = x^2
    x = torch.tensor([2.0], requires_grad=True)

    # Initialize the optimizer
    optimizer = BelavkinOptimizer([x], eta=0.1, gamma=0.0, beta=0.0) # SGD-like

    # Run a few optimization steps
    for _ in range(100):
        optimizer.zero_grad()
        loss = x.pow(2)
        loss.backward()
        optimizer.step()

    # The minimum is at x=0. Check if the optimizer has moved x closer to 0.
    assert torch.abs(x) < 0.1
