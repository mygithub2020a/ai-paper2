import torch
import torch.nn as nn
from belavkin.belopt.optim import BelOpt

def test_shape():
    model = nn.Linear(10, 1)
    optimizer = BelOpt(model.parameters())
    initial_shapes = [p.shape for p in model.parameters()]

    # Dummy training loop
    for _ in range(5):
        optimizer.zero_grad()
        output = model(torch.randn(1, 10))
        loss = output.sum()
        loss.backward()
        optimizer.step()

    final_shapes = [p.shape for p in model.parameters()]
    assert initial_shapes == final_shapes, "Parameter shapes changed after optimization."

if __name__ == "__main__":
    test_shape()
