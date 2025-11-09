import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, eta=1e-3, gamma=0.1, beta=0.01):
        if not 0.0 <= eta:
            raise ValueError(f"Invalid eta value: {eta}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(eta=eta, gamma=gamma, beta=beta)
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

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    pass

                gamma = group['gamma']
                eta = group['eta']
                beta = group['beta']

                # Belavkin update rule
                epsilon = torch.randn_like(grad)
                update = -(gamma * grad.pow(2) + eta * grad) + beta * grad * epsilon
                p.add_(update)

        return loss
