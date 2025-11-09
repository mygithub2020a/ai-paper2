import torch
import pytest
from belavkin.belopt.optim import BelOpt

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_compatibility():
    model = torch.nn.Linear(10, 1).cuda().half()
    optimizer = BelOpt(model.parameters())

    # Simulate a training step with mixed precision
    scaler = torch.cuda.amp.GradScaler()
    input_tensor = torch.randn(1, 10).cuda().half()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(input_tensor)
        loss = output.sum()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Check if parameters are still in FP16
    for p in model.parameters():
        assert p.dtype == torch.float16, "Parameter dtype changed after FP16 step"
