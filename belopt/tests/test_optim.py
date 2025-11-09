import torch
import unittest
from belopt.optim import BelOpt
from belopt.schedules import get_scheduler

class TestBelOpt(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.loss_fn = torch.nn.MSELoss()
        self.input_data = torch.randn(1, 10)
        self.target = torch.randn(1, 1)

    def test_optimizer_step(self):
        optimizer = BelOpt(self.model.parameters(), lr=0.1)

        optimizer.zero_grad()
        output = self.model(self.input_data)
        loss = self.loss_fn(output, self.target)
        loss.backward()
        optimizer.step()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

    def test_shape_dtype(self):
        initial_params = [p.clone() for p in self.model.parameters()]
        optimizer = BelOpt(self.model.parameters(), lr=0.1)

        optimizer.zero_grad()
        output = self.model(self.input_data)
        loss = self.loss_fn(output, self.target)
        loss.backward()
        optimizer.step()

        for p_initial, p_updated in zip(initial_params, self.model.parameters()):
            self.assertEqual(p_initial.shape, p_updated.shape)
            self.assertEqual(p_initial.dtype, p_updated.dtype)

    def test_determinism(self):
        torch.manual_seed(42)
        model1 = torch.nn.Linear(10, 1)
        optimizer1 = BelOpt(model1.parameters(), lr=0.1)

        optimizer1.zero_grad()
        output1 = model1(self.input_data)
        loss1 = self.loss_fn(output1, self.target)
        loss1.backward()
        optimizer1.step()
        params1 = [p.clone() for p in model1.parameters()]

        torch.manual_seed(42)
        model2 = torch.nn.Linear(10, 1)
        optimizer2 = BelOpt(model2.parameters(), lr=0.1)

        optimizer2.zero_grad()
        output2 = model2(self.input_data)
        loss2 = self.loss_fn(output2, self.target)
        loss2.backward()
        optimizer2.step()
        params2 = [p.clone() for p in model2.parameters()]

        for p1, p2 in zip(params1, params2):
            self.assertTrue(torch.equal(p1, p2))

    def test_fp16_safety(self):
        if not torch.cuda.is_available():
            self.skipTest("FP16 test requires a GPU")

        model_fp16 = torch.nn.Linear(10, 1).cuda().half()
        optimizer = BelOpt(model_fp16.parameters(), lr=0.1)
        input_data_fp16 = self.input_data.cuda().half()
        target_fp16 = self.target.cuda().half()

        optimizer.zero_grad()
        output = model_fp16(input_data_fp16)
        loss = self.loss_fn(output, target_fp16)
        loss.backward()
        optimizer.step()

        for param in model_fp16.parameters():
            self.assertFalse(torch.isnan(param).any())
            self.assertFalse(torch.isinf(param).any())

    def test_scheduler(self):
        optimizer = BelOpt(self.model.parameters(), lr=0.1, gamma=0.01, beta=0.001)

        lr_scheduler = get_scheduler(optimizer, 'lr', 'cosine', T_max=10)
        gamma_scheduler = get_scheduler(optimizer, 'gamma', 'cosine', T_max=10)
        beta_scheduler = get_scheduler(optimizer, 'beta', 'cosine', T_max=10)

        initial_lr = optimizer.param_groups[0]['lr']
        initial_gamma = optimizer.param_groups[0]['gamma']
        initial_beta = optimizer.param_groups[0]['beta']

        # First step: values should not change
        lr_scheduler.step()
        gamma_scheduler.step()
        beta_scheduler.step()

        self.assertEqual(initial_lr, optimizer.param_groups[0]['lr'])
        self.assertEqual(initial_gamma, optimizer.param_groups[0]['gamma'])
        self.assertEqual(initial_beta, optimizer.param_groups[0]['beta'])

        # Second step: values should change
        lr_scheduler.step()
        gamma_scheduler.step()
        beta_scheduler.step()

        self.assertNotEqual(initial_lr, optimizer.param_groups[0]['lr'])
        self.assertNotEqual(initial_gamma, optimizer.param_groups[0]['gamma'])
        self.assertNotEqual(initial_beta, optimizer.param_groups[0]['beta'])


if __name__ == '__main__':
    unittest.main()
