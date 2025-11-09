import torch
import torch.nn as nn
from optimizer import BelOptim, BelOptimSecondOrder

def test_optimizer(optimizer_class):
    # Simple model and data
    model = nn.Linear(10, 1)
    # Set model parameters to a fixed value
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.1)

    initial_params = [p.clone() for p in model.parameters()]

    optimizer = optimizer_class(model.parameters())

    # Dummy training loop
    for _ in range(5):
        optimizer.m = [torch.zeros_like(p) for p in model.parameters()] # Reset momentum
        input_data = torch.randn(5, 10)
        target = torch.randn(5, 1)

        # Forward pass
        output = model(input_data)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Get gradients
        gradients = [p.grad.clone() for p in model.parameters()]

        # Optimizer step
        optimizer.step(gradients)

    # Check if parameters have been updated
    updated_params = [p.clone() for p in model.parameters()]

    params_changed = False
    for initial, updated in zip(initial_params, updated_params):
        if not torch.equal(initial, updated):
            params_changed = True
            break

    return params_changed

def run_tests():
    beloptim_passed = test_optimizer(BelOptim)
    beloptim_so_passed = test_optimizer(BelOptimSecondOrder)

    print(f"BelOptim test passed: {beloptim_passed}")
    print(f"BelOptimSecondOrder test passed: {beloptim_so_passed}")

if __name__ == "__main__":
    run_tests()
