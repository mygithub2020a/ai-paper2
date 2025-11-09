import unittest
import torch
import torch.nn as nn
from beloptim.optimizers.belavkin import BelOptim
from beloptim.optimizers.belavkin_variants import BelOptimWithMomentum, BelOptimAdaptive, BelOptimLayerwise

class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
        self.input = torch.randn(5, 10)
        self.target = torch.randn(5, 1)
        self.loss_fn = nn.MSELoss()

    def _run_optimizer_test(self, optimizer_class, **kwargs):
        optimizer = optimizer_class(self.model.parameters(), **kwargs)

        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]

        # Perform one optimization step
        optimizer.zero_grad()
        output = self.model(self.input)
        loss = self.loss_fn(output, self.target)
        loss.backward()
        optimizer.step()

        # Check if parameters have been updated
        for i, p in enumerate(self.model.parameters()):
            self.assertFalse(torch.equal(initial_params[i], p.data))

    def test_beloptim(self):
        self._run_optimizer_test(BelOptim)

    def test_beloptim_with_momentum(self):
        self._run_optimizer_test(BelOptimWithMomentum)

    def test_beloptim_adaptive(self):
        self._run_optimizer_test(BelOptimAdaptive)

    def test_beloptim_layerwise(self):
        # Layerwise requires parameter groups
        param_groups = [
            {'params': list(self.model.children())[0].parameters(), 'gamma_base': 0.05, 'beta_base': 0.005},
            {'params': list(self.model.children())[2].parameters(), 'gamma_base': 0.1, 'beta_base': 0.01}
        ]
        optimizer = BelOptimLayerwise(param_groups)

        initial_params = [p.clone() for p in self.model.parameters()]

        optimizer.zero_grad()
        output = self.model(self.input)
        loss = self.loss_fn(output, self.target)
        loss.backward()
        optimizer.step()

        for i, p in enumerate(self.model.parameters()):
            self.assertFalse(torch.equal(initial_params[i], p.data))

if __name__ == '__main__':
    unittest.main()
