import torch
from belavkin.optimizer import BelavkinOptimizer

def test_optimizer_step():
    model = torch.nn.Linear(2, 1)
    initial_params = [p.clone() for p in model.parameters()]
    optimizer = BelavkinOptimizer(model.parameters())

    # Create a dummy loss and gradients
    loss = model(torch.randn(1, 2)).sum()
    loss.backward()

    # Take a step
    optimizer.step()

    # Check that the parameters have changed
    for p_initial, p_final in zip(initial_params, model.parameters()):
        assert not torch.equal(p_initial, p_final)
