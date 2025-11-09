import torch

class BelavkinOptimizer(torch.optim.Optimizer):
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
                 adaptive_gamma=False, adaptive_beta=False, alpha=0.5, ema_decay=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta,
                        alpha=alpha, ema_decay=ema_decay)
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
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                gamma = group['gamma']
                lr = group['lr']
                beta = group['beta']
                alpha = group['alpha']
                ema_decay = group['ema_decay']

                # Assuming dt = 1 for simplicity
                dt = 1

                if group['adaptive_gamma']:
                    grad_norm_sq = grad.pow(2).sum()
                    gamma = gamma * (1 + grad_norm_sq) ** (-alpha)

                if group['adaptive_beta']:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(ema_decay).addcmul_(grad, grad, value=1 - ema_decay)
                    beta = beta / (1 + exp_avg_sq.sqrt())


                # Damping term
                damping = gamma * grad.pow(2) * dt

                # Drift term
                drift = lr * grad * dt

                # Stochastic term
                noise = torch.randn_like(grad)
                stochastic = beta * grad * (dt**0.5) * noise

                p.data.add_(- (damping + drift) + stochastic)

        return loss
