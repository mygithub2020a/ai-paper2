import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, eta=0.01, gamma=0.1, beta=0.01, clip_value=1.0):
        if not 0.0 <= eta:
            raise ValueError(f"Invalid learning rate: {eta}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid adaptive damping factor: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid stochastic exploration factor: {beta}")
        if not 0.0 <= clip_value:
            raise ValueError(f"Invalid clip value: {clip_value}")

        defaults = dict(eta=eta, gamma=gamma, beta=beta, clip_value=clip_value)
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

                eta = group['eta']
                gamma = group['gamma']
                beta = group['beta']
                clip_value = group['clip_value']

                # Clip the gradients
                grad.clamp_(-clip_value, clip_value)

                # Belavkin update rule
                # dθ = -[γ * (∇L(θ))^2 + η * ∇L(θ)] + β * ∇L(θ) * ε

                # First term: -[γ * (∇L(θ))^2 + η * ∇L(θ)]
                damped_grad = gamma * grad.pow(2) + eta * grad

                # Second term: β * ∇L(θ) * ε
                noise = torch.randn_like(grad)
                stochastic_term = beta * grad * noise

                # Update parameters
                p.add_(-damped_grad + stochastic_term)

        return loss
