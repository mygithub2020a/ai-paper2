import torch
import torch.nn as nn
from belavkin.belopt.optim import BelOpt
import copy

def test_determinism():
    # --- First run ---
    torch.manual_seed(0)
    model1 = nn.Linear(10, 1)
    # Use a non-zero beta to test the stochastic path
    optimizer1 = BelOpt(model1.parameters(), beta0=0.1)

    for _ in range(5):
        data = torch.randn(1, 10)
        optimizer1.zero_grad()
        output1 = model1(data)
        loss1 = output1.sum()
        loss1.backward()
        optimizer1.step()

    # --- Second run ---
    torch.manual_seed(0) # Reset the seed
    model2 = nn.Linear(10, 1)
    optimizer2 = BelOpt(model2.parameters(), beta0=0.1)

    for _ in range(5):
        data = torch.randn(1, 10)
        optimizer2.zero_grad()
        output2 = model2(data)
        loss2 = output2.sum()
        loss2.backward()
        optimizer2.step()

    # Check that the parameters of the two models are identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Optimizer is not deterministic."

if __name__ == "__main__":
    test_determinism()
