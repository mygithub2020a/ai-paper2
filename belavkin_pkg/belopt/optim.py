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
        betas (Tuple[float, float]): Coefficients for moving averages (default: (0.9, 0.999))
        alpha (float): Exponent for adaptive gamma calculation (default: 0.5)
        clip_grad_norm (float): Max norm of the gradients (default: None)
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2,
                 adaptive_gamma=False, adaptive_beta=False,
                 betas=(0.9, 0.999), alpha=0.5, clip_grad_norm=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if clip_grad_norm is not None and not 0.0 <= clip_grad_norm:
            raise ValueError(f"Invalid clip_grad_norm value: {clip_grad_norm}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta,
                        betas=betas, alpha=alpha, clip_grad_norm=clip_grad_norm)
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
            if group['clip_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], group['clip_grad_norm'])

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
                    # Moving averages of gradient and its square
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']

                if group['adaptive_gamma']:
                    alpha = group['alpha']
                    gamma = gamma * (1 + corrected_exp_avg_sq).pow(-alpha)

                if group['adaptive_beta']:
                    variance = torch.clamp(corrected_exp_avg_sq - corrected_exp_avg.pow(2), min=0)
                    beta = beta * torch.exp(-variance)

                # Update rule
                damping = gamma * grad.pow(2)
                drift = lr * grad

                noise = torch.randn_like(grad)
                diffusion = beta * grad * noise

                p.data.add_(- (damping + drift) + diffusion)

        return loss
