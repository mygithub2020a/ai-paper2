import torch
from torch.optim.optimizer import Optimizer, required

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, eta=required, gamma=0.1, beta=0.01, gamma_decay=0.0):
        if eta is not required and eta < 0.0:
            raise ValueError(f"Invalid learning rate: {eta}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if gamma_decay < 0.0:
            raise ValueError(f"Invalid gamma_decay value: {gamma_decay}")

        defaults = dict(eta=eta, gamma=gamma, beta=beta, gamma_decay=gamma_decay)
        super(BelavkinOptimizer, self).__init__(params, defaults)

        # Initialize step counter
        for group in self.param_groups:
            group['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group['step'] += 1
            eta = group['eta']
            gamma = group['gamma']
            beta = group['beta']
            gamma_decay = group['gamma_decay']

            # Apply gamma decay
            if gamma_decay > 0:
                gamma = gamma / (1 + group['step'] * gamma_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                # dθ = -[γ ⊙ (∇L(θ))² + η ∇L(θ)] + β ∇L(θ) ⊙ ε
                term1 = - (gamma * grad.pow(2) + eta * grad)

                epsilon = torch.randn_like(grad)
                term2 = beta * grad * epsilon

                d_theta = term1 + term2

                p.add_(d_theta)

        return loss
