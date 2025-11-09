import torch
from belavkin.belopt.optim import BelOpt

def test_param_shapes():
    model = torch.nn.Linear(10, 1)
    optimizer = BelOpt(model.parameters())

    initial_shapes = [p.shape for p in model.parameters()]

    # Simulate a training step
    optimizer.zero_grad()
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    optimizer.step()

    final_shapes = [p.shape for p in model.parameters()]

    assert initial_shapes == final_shapes, "Parameter shapes changed after optimizer step"
