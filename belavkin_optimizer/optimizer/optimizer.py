import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=1e-3, beta=1e-3):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BelavkinOptimizer does not support sparse gradients")

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']

                # Generate isotropic Gaussian noise
                epsilon = torch.randn_like(p)

                # Belavkin optimizer update rule
                update = -(gamma * grad.pow(2) + lr * grad) + beta * grad * epsilon

                p.add_(update)

        return loss
