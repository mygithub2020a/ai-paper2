import torch
from belavkin.belopt.optim import BelOpt

def test_determinism():
    torch.manual_seed(0)
    model1 = torch.nn.Linear(10, 1)
    optimizer1 = BelOpt(model1.parameters())

    torch.manual_seed(0)
    model2 = torch.nn.Linear(10, 1)
    optimizer2 = BelOpt(model2.parameters())

    # First step
    input_tensor = torch.randn(1, 10)
    optimizer1.zero_grad()
    loss1 = model1(input_tensor).sum()
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2 = model2(input_tensor).sum()
    loss2.backward()
    optimizer2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Models are not deterministic after one step"

    # Second step
    input_tensor = torch.randn(1, 10)
    optimizer1.zero_grad()
    loss1 = model1(input_tensor).sum()
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2 = model2(input_tensor).sum()
    loss2.backward()
    optimizer2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Models are not deterministic after two steps"
