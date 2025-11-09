import torch
from torch import nn
import unittest
import math

from belavkin_pkg.belopt.optim import BelavkinOptimizer

class TestBelavkinOptimizer(unittest.TestCase):
    def test_optimizer_step(self):
        model = nn.Linear(10, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1)

        # Save initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Dummy forward and backward pass
        input_tensor = torch.randn(1, 10)
        target = torch.randn(1, 1)
        output = model(input_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        optimizer.step()

        # Check that parameters have been updated
        for i, p in enumerate(model.parameters()):
            self.assertFalse(torch.equal(p, initial_params[i]))

    def test_update_rule(self):
        param = nn.Parameter(torch.tensor([1.0]))
        optimizer = BelavkinOptimizer([param], lr=0.1, gamma=0.01, beta=0.0)

        # Dummy gradient
        grad = torch.tensor([2.0])
        param.grad = grad

        # Expected update
        damping = 0.01 * grad.pow(2)
        drift = 0.1 * grad
        expected_param = param.data.clone() - (damping + drift)

        optimizer.step()

        self.assertTrue(torch.allclose(param.data, expected_param))

    def test_adaptive_gamma(self):
        param = nn.Parameter(torch.tensor([1.0]))
        initial_param = param.data.clone()
        optimizer = BelavkinOptimizer([param], lr=0.1, gamma=0.01, beta=0.0, adaptive_gamma=True, alpha=0.5)

        # Dummy gradient
        grad = torch.tensor([2.0])
        param.grad = grad

        # Expected calculation for the first step
        beta2 = optimizer.param_groups[0]['betas'][1]
        exp_avg_sq = (1-beta2) * grad.pow(2)
        bias_correction2 = 1 - beta2 ** 1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2
        gamma = 0.01 * (1 + corrected_exp_avg_sq).pow(-0.5)

        # Expected update
        damping = gamma * grad.pow(2)
        drift = 0.1 * grad
        expected_param = initial_param - (damping + drift)

        optimizer.step()

        self.assertTrue(torch.allclose(param.data, expected_param))


if __name__ == '__main__':
    unittest.main()
