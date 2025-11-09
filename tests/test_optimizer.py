import torch
import unittest
import copy
from belavkin.optim.belavkin import Belavkin

class TestBelavkin(unittest.TestCase):
    def test_update_rule(self):
        param = torch.randn(10, 10, requires_grad=True)
        optimizer = Belavkin([param])
        param.grad = torch.randn(10, 10)
        p_orig = param.clone().detach()
        optimizer.step()
        self.assertFalse(torch.equal(p_orig, param))

    def test_state_dict(self):
        # Use a deterministic optimizer config (no noise) to test state loading
        p1 = torch.randn(10, 10, requires_grad=True)
        opt1 = Belavkin([p1], beta=0)

        # Run for a few steps to build up internal state (m2)
        for _ in range(5):
            p1.grad = torch.randn(10, 10)
            opt1.step()

        # Save the state dict from the first optimizer
        state_dict = opt1.state_dict()
        p1_after_5_steps = p1.clone().detach()

        # Create a second optimizer to simulate resuming training
        p2 = p1_after_5_steps.clone().detach().requires_grad_(True)
        opt2 = Belavkin([p2], beta=0)

        # Load a deepcopy of the state to prevent shared state tensors
        opt2.load_state_dict(copy.deepcopy(state_dict))

        # Check that the internal state was loaded correctly
        self.assertTrue(torch.equal(opt1.state[p1]['m2'], opt2.state[p2]['m2']))

        # Use the same gradient for both optimizers for the next step
        grad = torch.randn(10, 10)
        p1.grad, p2.grad = grad.clone(), grad.clone()

        opt1.step()
        opt2.step()

        # After one more step, the parameters should be identical
        self.assertTrue(torch.equal(p1, p2))

    def test_deterministic_seeding(self):
        # First run
        torch.manual_seed(42)
        param1 = torch.randn(10, 10, requires_grad=True)
        optimizer1 = Belavkin([param1])
        for _ in range(5):
            param1.grad = torch.randn(10, 10)
            optimizer1.step()

        # Second run with the same seed
        torch.manual_seed(42)
        param2 = torch.randn(10, 10, requires_grad=True)
        optimizer2 = Belavkin([param2])
        for _ in range(5):
            param2.grad = torch.randn(10, 10)
            optimizer2.step()

        self.assertTrue(torch.equal(param1, param2))

    def test_gradient_clipping(self):
        clip_value = 1.0
        param = torch.randn(10, 10, requires_grad=True)
        optimizer = Belavkin([param], grad_clip=clip_value)

        # Assign a gradient with a norm larger than the clip value
        grad = torch.randn(10, 10)
        grad_norm = torch.norm(grad)
        param.grad = grad / grad_norm * (clip_value + 1.0)

        # The step function will clip the gradient in-place before the update
        optimizer.step()

        # Check that the norm of the gradient is now approximately the clip value
        self.assertLessEqual(torch.norm(param.grad), clip_value)

    def test_grad_scaled_noise(self):
        param = torch.randn(10, 10, requires_grad=True)
        grad = torch.randn(10, 10)

        # Optimizer with standard noise
        torch.manual_seed(123)
        param1 = param.clone().detach().requires_grad_(True)
        param1.grad = grad.clone()
        optimizer_normal_noise = Belavkin([param1], grad_scaled_noise=False)
        optimizer_normal_noise.step()

        # Optimizer with gradient-scaled noise
        torch.manual_seed(123)
        param2 = param.clone().detach().requires_grad_(True)
        param2.grad = grad.clone()
        optimizer_scaled_noise = Belavkin([param2], grad_scaled_noise=True)
        optimizer_scaled_noise.step()

        # The parameter updates should be different
        self.assertFalse(torch.equal(param1, param2))

if __name__ == '__main__':
    unittest.main()
