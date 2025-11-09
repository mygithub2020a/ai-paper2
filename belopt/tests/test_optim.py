import torch
import unittest
from belopt.optim import BelOpt
from belopt.schedules import ConstantScheduler

class TestBelOpt(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 10)
        self.target = torch.randn(1, 1)
        self.loss_fn = torch.nn.MSELoss()

    def _run_step(self, model, optimizer):
        optimizer.zero_grad()
        output = model(self.input_tensor.to(model.weight.dtype))
        loss = self.loss_fn(output, self.target.to(model.weight.dtype))
        loss.backward()
        optimizer.step()

    def test_optimizer_init(self):
        model = torch.nn.Linear(10, 1)
        optimizer = BelOpt(model.parameters())
        self.assertIsInstance(optimizer, BelOpt)

    def test_optimizer_step(self):
        model = torch.nn.Linear(10, 1)
        optimizer = BelOpt(model.parameters())
        initial_params = [p.clone() for p in model.parameters()]

        self._run_step(model, optimizer)

        for i, p in enumerate(model.parameters()):
            self.assertFalse(torch.equal(p, initial_params[i]))

    def test_determinism(self):
        # Run 1
        torch.manual_seed(0)
        model1 = torch.nn.Linear(10, 1)
        optimizer1 = BelOpt(model1.parameters(), beta_scheduler=ConstantScheduler(0.1))
        self._run_step(model1, optimizer1)

        # Run 2 - should be identical
        torch.manual_seed(0)
        model2 = torch.nn.Linear(10, 1)
        optimizer2 = BelOpt(model2.parameters(), beta_scheduler=ConstantScheduler(0.1))
        self._run_step(model2, optimizer2)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_dtype_consistency(self):
        model = torch.nn.Linear(10, 1, dtype=torch.float64)
        optimizer = BelOpt(model.parameters())
        self._run_step(model, optimizer)

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                self.assertEqual(state['exp_avg_sq'].dtype, torch.float64)

    def test_fp16_safety(self):
        if not torch.cuda.is_available():
            self.skipTest("Skipping FP16 test, CUDA not available")
            return

        model = torch.nn.Linear(10, 1).cuda().half()
        optimizer = BelOpt(model.parameters())

        # Use a local input tensor on the correct device and dtype
        input_tensor_fp16 = self.input_tensor.cuda().half()
        target_fp16 = self.target.cuda().half()

        optimizer.zero_grad()
        output = model(input_tensor_fp16)
        loss = self.loss_fn(output, target_fp16)
        loss.backward()
        optimizer.step()

        for group in optimizer.param_groups:
            for p in group['params']:
                self.assertEqual(p.dtype, torch.float16)
                state = optimizer.state[p]
                self.assertEqual(state['exp_avg_sq'].dtype, torch.float16)

if __name__ == '__main__':
    unittest.main()
