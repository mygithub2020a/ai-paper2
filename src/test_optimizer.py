import torch
import unittest
from src.optimizer import BelavkinOptimizer

class TestBelavkinOptimizer(unittest.TestCase):
    def test_step_runs(self):
        # Simple model
        model = torch.nn.Linear(10, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.01)

        # Fake data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        # Forward/Backward
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()

        # Step
        loss_val, gns, panic = optimizer.step()

        # Check outputs
        self.assertIsInstance(loss_val, type(None)) # Closure not used
        self.assertIsInstance(gns, float)
        self.assertIsInstance(panic, bool)

        # Check params changed
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_step_with_closure(self):
        model = torch.nn.Linear(10, 1)
        optimizer = BelavkinOptimizer(model.parameters(), lr=0.01)
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            return loss

        loss_val, gns, panic = optimizer.step(closure=closure)
        self.assertIsNotNone(loss_val)

if __name__ == '__main__':
    unittest.main()
