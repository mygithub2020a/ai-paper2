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
        adaptive_beta (bool): Adapt β based on loss landscape curvature (Not implemented)
        alpha (float): Exponent for adaptive damping (default: 1.0)
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2,
                 adaptive_gamma=False, adaptive_beta=False, alpha=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta, alpha=alpha)
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
            if group['adaptive_beta']:
                raise NotImplementedError('adaptive_beta is not implemented yet')

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']
                adaptive_gamma = group['adaptive_gamma']
                alpha = group['alpha']

                current_gamma = gamma
                if adaptive_gamma:
                    grad_norm_sq = torch.sum(grad ** 2)
                    current_gamma = gamma / ((1 + grad_norm_sq) ** alpha)

                # Damping term
                damping = current_gamma * (grad ** 2)

                # Drift term
                drift = lr * grad

                # Stochastic exploration term
                noise = torch.randn_like(grad)
                exploration = beta * grad * noise

                # Update rule
                p.data.add_(- (damping + drift) + exploration)

        return loss
