import torch
import unittest
from belavkin.optimizer import BelavkinOptimizer

class TestBelavkinOptimizer(unittest.TestCase):
    def test_optimizer_step(self):
        # A simple test case to ensure the optimizer runs without errors
        model = torch.nn.Linear(2, 1)
        model.weight.data = torch.tensor([[0.5, 0.5]])
        model.bias.data = torch.tensor([0.1])
        initial_params = [p.clone() for p in model.parameters()]

        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, gamma=0.01, beta=0.0)
        loss_fn = torch.nn.MSELoss()

        input = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0]])

        optimizer.zero_grad()
        output = model(input) # 0.5*1 + 0.5*2 + 0.1 = 1.6
        loss = loss_fn(output, target) # (1.6 - 1.0)^2 = 0.36
        loss.backward()
        # dL/dw = 2 * (output - target) * input = 2 * 0.6 * [1, 2] = [1.2, 2.4]
        # dL/db = 2 * (output - target) = 1.2

        optimizer.step()

        # w_new = w_old - (gamma * grad^2 + lr * grad)
        # w1_new = 0.5 - (0.01 * 1.2^2 + 0.1 * 1.2) = 0.5 - (0.0144 + 0.12) = 0.3656
        # w2_new = 0.5 - (0.01 * 2.4^2 + 0.1 * 2.4) = 0.5 - (0.0576 + 0.24) = 0.2024
        # b_new = 0.1 - (0.01 * 1.2^2 + 0.1 * 1.2) = 0.1 - (0.0144 + 0.12) = -0.0344

        self.assertAlmostEqual(model.weight.data[0, 0].item(), 0.3656, places=4)
        self.assertAlmostEqual(model.weight.data[0, 1].item(), 0.2024, places=4)
        self.assertAlmostEqual(model.bias.data[0].item(), -0.0344, places=4)

    def test_adaptive_gamma(self):
        # Test that adaptive gamma is calculated correctly
        model = torch.nn.Linear(2, 1)
        model.weight.data = torch.tensor([[0.5, 0.5]])
        model.bias.data = torch.tensor([0.1])

        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, gamma=1.0, beta=0.0, adaptive_gamma=True)
        loss_fn = torch.nn.MSELoss()

        input = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0]])

        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        # total_grad_norm_sq = 1.2**2 + 2.4**2 + 1.2**2 = 1.44 + 5.76 + 1.44 = 8.64
        # gamma_eff = 1.0 / (1 + 8.64) = 1/9.64 = 0.1037344...
        # w1_update = gamma_eff * 1.2**2 + 0.1 * 1.2 = 0.269377...
        # w1_new = 0.5 - 0.269377 = 0.230622...
        # w2_update = gamma_eff * 2.4**2 + 0.1 * 2.4 = 0.837510...
        # w2_new = 0.5 - 0.837510 = -0.33751...
        # b_update = gamma_eff * 1.2**2 + 0.1 * 1.2 = 0.269377...
        # b_new = 0.1 - 0.269377 = -0.169377...

        optimizer.step()

        self.assertAlmostEqual(model.weight.data[0, 0].item(), 0.230622, places=5)
        self.assertAlmostEqual(model.weight.data[0, 1].item(), -0.33751, places=5)
        self.assertAlmostEqual(model.bias.data[0].item(), -0.169377, places=5)

    def test_stochastic_behavior(self):
        # Test that the optimizer is stochastic when beta > 0
        model = torch.nn.Linear(10, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, beta=0.1)
        loss_fn = torch.nn.MSELoss()

        input = torch.randn(1, 10)
        target = torch.randn(1, 1)

        # Run 1
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        params1 = [p.clone() for p in model.parameters()]

        # Run 2
        model = torch.nn.Linear(10, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, beta=0.1)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        params2 = [p.clone() for p in model.parameters()]

        # Check that the parameters are different
        for i, p in enumerate(params1):
            self.assertFalse(torch.equal(p, params2[i]))

if __name__ == '__main__':
    unittest.main()
