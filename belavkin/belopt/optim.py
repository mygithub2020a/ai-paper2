import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    Args:
        params: Model parameters
        lr (float): Learning rate η (default: 1e-3)
        gamma (float): Damping factor γ (default: 1e-4)
        beta (float): Exploration factor β (default: 1e-2)
        adaptive_gamma (bool): Adapt γ based on gradient statistics
        adaptive_beta (bool): Adapt β based on loss landscape curvature
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2,
                 adaptive_gamma=False, adaptive_beta=False, alpha=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta,
                        alpha=alpha)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Implements: θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_norm_sq_avg'] = torch.zeros_like(p.data)

                state['step'] += 1
                grad_norm_sq_avg = state['grad_norm_sq_avg']
                beta1 = 0.9 # A default value for the running average, can be parameterized
                grad_norm_sq_avg.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']
                adaptive_gamma = group['adaptive_gamma']
                adaptive_beta = group['adaptive_beta'] # Not used yet, will be added later
                alpha = group['alpha']

                if adaptive_gamma:
                    gamma = gamma / (1 + grad_norm_sq_avg).pow(alpha)

                # Update rule
                damping = gamma * grad.pow(2)
                drift = lr * grad
                noise = beta * grad * torch.randn_like(grad)

                p.data.add_(- (damping + drift) + noise)

        return loss
