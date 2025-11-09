import torch
import unittest
from belavkin.belopt.optim import BelavkinOptimizer

class TestBelavkinOptimizer(unittest.TestCase):
    def test_optimizer_step(self):
        # Create a simple model and some dummy data
        model = torch.nn.Linear(2, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.1, gamma=0.01, beta=0.01)
        criterion = torch.nn.MSELoss()
        input = torch.randn(1, 2)
        target = torch.randn(1, 1)

        # Take a step and check if the parameters have been updated
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step()

        for i, p in enumerate(model.parameters()):
            self.assertFalse(torch.equal(p.data, initial_params[i].data))

    def test_convergence(self):
        # Create a simple model and some dummy data
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(1.0)

        optimizer = BelavkinOptimizer(model.parameters(), lr=0.01, gamma=0.0, beta=0.0)
        criterion = torch.nn.MSELoss()

        # y = 3x + 2
        input = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        target = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

        for _ in range(1000):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        self.assertAlmostEqual(model.weight.item(), 3.0, places=1)
        self.assertAlmostEqual(model.bias.item(), 2.0, places=1)


if __name__ == '__main__':
    unittest.main()
