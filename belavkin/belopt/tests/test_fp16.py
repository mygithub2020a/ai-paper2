import torch
import torch.nn as nn
from belavkin.belopt.optim import BelOpt

def test_fp16():
    model = nn.Linear(10, 1).half() # Use a half-precision model
    optimizer = BelOpt(model.parameters())

    # Dummy training loop
    for _ in range(5):
        optimizer.zero_grad()
        # Use half-precision data
        data = torch.randn(1, 10).half()
        output = model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    # Check that all parameters are still half-precision
    for p in model.parameters():
        assert p.dtype == torch.float16, f"Parameter dtype changed to {p.dtype}."

    # Check for NaNs
    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaNs found in parameters after optimization."

if __name__ == "__main__":
    test_fp16()
