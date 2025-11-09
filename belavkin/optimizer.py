import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, gamma=0.01, eta=0.01, beta=0.01):
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= eta:
            raise ValueError(f"Invalid eta value: {eta}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(gamma=gamma, eta=eta, beta=beta)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            eta = group['eta']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BelavkinOptimizer does not support sparse gradients")

                # Core update rule
                d_p = - (gamma * grad.pow(2) + eta * grad) + beta * grad * torch.randn_like(grad)
                p.add_(d_p)

        return loss
