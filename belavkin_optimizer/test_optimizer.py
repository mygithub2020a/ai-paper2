import torch
import unittest
from belavkin_optimizer.optimizer import BelavkinOptimizer

class TestBelavkinOptimizer(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        # Initialize weights to a known value for reproducibility
        torch.nn.init.constant_(self.model.weight, 0.5)
        torch.nn.init.constant_(self.model.bias, 0.1)
        self.input_tensor = torch.ones(1, 10) # Use ones for predictable gradients
        self.target = torch.zeros(1, 1)

    def _get_grad(self, loss_scale=1.0):
        self.model.zero_grad()
        output = self.model(self.input_tensor)
        loss = (output - self.target).pow(2).sum() * loss_scale
        loss.backward()
        return [p.grad.clone() for p in self.model.parameters()]

    def test_basic_step(self):
        optimizer = BelavkinOptimizer(self.model.parameters(), lr=0.01, gamma=0.001, beta=0)

        initial_params = [p.clone() for p in self.model.parameters()]
        grads = self._get_grad()
        optimizer.step()

        for i, p in enumerate(self.model.parameters()):
            # Expected update: p_new = p_old - (gamma * grad^2 + lr * grad)
            expected_update = initial_params[i] - (0.001 * grads[i].pow(2) + 0.01 * grads[i])
            self.assertTrue(torch.allclose(p, expected_update))

    def test_adaptive_gamma(self):
        optimizer = BelavkinOptimizer(self.model.parameters(), lr=0.01, gamma=0.001, beta=0, adaptive_gamma=True, alpha=0.5)

        initial_params = [p.clone() for p in self.model.parameters()]
        grads = self._get_grad()

        global_grad_norm_sq = sum(torch.sum(g * g) for g in grads)
        gamma_t = 0.001 * (1 + global_grad_norm_sq).pow(-0.5)

        optimizer.step()

        for i, p in enumerate(self.model.parameters()):
            expected_update = initial_params[i] - (gamma_t * grads[i].pow(2) + 0.01 * grads[i])
            self.assertTrue(torch.allclose(p, expected_update, atol=1e-6))

    def test_stochasticity(self):
        # We can't test the exact stochastic term, but we can test that it's applied
        # and that two steps with the same gradient but different seeds are different.
        optimizer = BelavkinOptimizer(self.model.parameters(), lr=0.01, gamma=0.001, beta=0.1)

        # First step
        torch.manual_seed(0)
        self._get_grad()
        optimizer.step()
        params_after_step1 = [p.clone() for p in self.model.parameters()]

        # Reset model and optimizer
        self.setUp()
        optimizer = BelavkinOptimizer(self.model.parameters(), lr=0.01, gamma=0.001, beta=0.1)

        # Second step with different seed
        torch.manual_seed(1)
        self._get_grad()
        optimizer.step()
        params_after_step2 = [p.clone() for p in self.model.parameters()]

        # The parameters should be different due to the different random noise
        params_are_different = any(not torch.equal(p1, p2) for p1, p2 in zip(params_after_step1, params_after_step2))
        self.assertTrue(params_are_different)

    def test_gradient_clipping(self):
        max_norm = 1.0
        optimizer = BelavkinOptimizer(self.model.parameters(), lr=0.01, clip_grad_norm=max_norm)

        # Generate gradients with a norm larger than max_norm
        self._get_grad(loss_scale=100.0)

        params_with_grad = [p for p in self.model.parameters() if p.grad is not None]

        # Calculate original norm
        original_total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2)
        self.assertGreater(original_total_norm.item(), max_norm)

        optimizer.step()

        # The optimizer clips gradients in-place, so we check the norm of the .grad attributes after the step
        clipped_total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2)

        self.assertAlmostEqual(clipped_total_norm.item(), max_norm, places=5)

if __name__ == '__main__':
    unittest.main()
