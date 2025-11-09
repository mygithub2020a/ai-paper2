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
                 adaptive_gamma=False, adaptive_beta=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta)
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
                    # For adaptive gamma, we'll store a running average of the squared gradient
                    if group['adaptive_gamma']:
                        state['grad_norm_sq_avg'] = torch.zeros_like(p.data)
                    if group['adaptive_beta']:
                        state['prev_grad'] = torch.zeros_like(p.data)

                state['step'] += 1

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']

                # Adaptive damping
                if group['adaptive_gamma']:
                    # A simple adaptive rule for gamma: γ_t = γ_0 / (1 + ||∇L||²)
                    # We use a running average for ||∇L||² for stability.
                    alpha = 0.5 # A reasonable default for the exponent
                    grad_norm_sq = grad * grad
                    exp_avg_sq = state['grad_norm_sq_avg']
                    # Using an exponential moving average for the squared gradient norm
                    ema_decay = 0.9
                    exp_avg_sq.mul_(ema_decay).add_(grad_norm_sq, alpha=1 - ema_decay)

                    gamma_t = gamma / ((1 + exp_avg_sq).pow(alpha))
                else:
                    gamma_t = gamma

                # Adaptive beta
                if group['adaptive_beta']:
                    prev_grad = state['prev_grad']
                    grad_diff = grad - prev_grad
                    # A simple rule for adapting beta: β_t = β_0 * (1 + ||∇L_t - ∇L_{t-1}||²)
                    # This increases beta when the gradient is changing rapidly (high curvature).
                    curvature_proxy = (grad_diff * grad_diff).sum()
                    beta_t = beta * (1 + curvature_proxy)
                    state['prev_grad'] = grad.clone()
                else:
                    beta_t = beta

                # Update rule: θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε
                # Assuming Δt = 1 for simplicity in discrete updates.

                # Deterministic part of the update
                deterministic_update = - (gamma_t * grad.pow(2) + lr * grad)

                # Stochastic part of the update
                if beta_t > 0:
                    noise = torch.randn_like(grad)
                    stochastic_update = beta_t * grad * noise
                else:
                    stochastic_update = 0.0

                # Combine and apply the update
                p.data.add_(deterministic_update + stochastic_update)

        return loss
