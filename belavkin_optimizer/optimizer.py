import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_

class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    Args:
        params: Model parameters
        lr (float): Learning rate η (default: 1e-3)
        gamma (float): Damping factor γ (default: 1e-4)
        beta (float): Exploration factor β (default: 1e-2)
        adaptive_gamma (bool): Adapt γ based on gradient statistics
        adaptive_beta (bool): Adapt β based on loss landscape curvature (Not Implemented)
        alpha (float): Exponent for adaptive gamma calculation (default: 0.5)
        clip_grad_norm (float): Max norm of the gradients (default: 0, no clipping)
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2,
                 adaptive_gamma=False, adaptive_beta=False, alpha=0.5, clip_grad_norm=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= clip_grad_norm:
            raise ValueError(f"Invalid clip_grad_norm value: {clip_grad_norm}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma,
                        adaptive_beta=adaptive_beta, alpha=alpha,
                        clip_grad_norm=clip_grad_norm)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Implements: θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε
        Assuming Δt = 1
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_grad_norm_value = self.param_groups[0]['clip_grad_norm']
        if clip_grad_norm_value > 0:
            all_params = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        all_params.append(p)
            if all_params:
                clip_grad_norm_(all_params, max_norm=clip_grad_norm_value)

        # Calculate global gradient norm for adaptive gamma
        global_grad_norm_sq = 0
        if self.param_groups[0]['adaptive_gamma']:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    global_grad_norm_sq += torch.sum(grad * grad)

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            adaptive_gamma = group['adaptive_gamma']
            adaptive_beta = group['adaptive_beta'] # Not implemented
            alpha = group['alpha']

            if adaptive_gamma:
                gamma_t = gamma * (1 + global_grad_norm_sq).pow(-alpha)
            else:
                gamma_t = gamma

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                # Update rule
                damping_term = gamma_t * grad.pow(2)
                drift_term = lr * grad

                stochastic_term = 0
                if beta > 0:
                    noise = torch.randn_like(grad)
                    stochastic_term = beta * grad * noise

                p.add_(- (damping_term + drift_term) + stochastic_term)

        return loss
