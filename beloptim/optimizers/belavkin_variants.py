import torch
from torch.optim.optimizer import Optimizer
import math

class BelOptimWithMomentum(Optimizer):
    """
    Belavkin Optimizer with Momentum.

    m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L(θ)
    v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L(θ))²
    θ_{t+1} = θ_t - [γ * v_t + η * m_t] + β * m_t * ε_t
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.1, beta=0.01,
                 noise_type='innovation', eps=1e-8, **kwargs):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas not in [0, 1): {betas}")

        defaults = dict(lr=lr, betas=betas, gamma=gamma, beta=beta,
                        noise_type=noise_type, eps=eps)
        super(BelOptimWithMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BelOptim does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L(θ)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L(θ))²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_t = exp_avg
                v_t = exp_avg_sq

                lr = group['lr']
                gamma = group['gamma']
                beta_noise = group['beta']
                noise_type = group['noise_type']

                noise = 0
                if beta_noise > 0:
                    if noise_type == 'gaussian':
                        noise = torch.randn_like(grad)
                    elif noise_type == 'innovation':
                        noise = torch.randn_like(grad)

                # θ_{t+1} = θ_t - [γ * v_t + η * m_t] + β * m_t * ε_t
                update = gamma * v_t + lr * m_t
                stochastic_term = beta_noise * m_t * noise

                p.data.add_(-update + stochastic_term)

        return loss

class BelOptimAdaptive(Optimizer):
    """
    Belavkin Optimizer with Adaptive Measurement Strength.

    γ_t = γ₀ * exp(-α * uncertainty_t)
    uncertainty_t = running_variance(∇L(θ))
    """
    def __init__(self, params, lr=1e-3, gamma0=0.1, beta=0.01, alpha=0.1,
                 momentum=0.9, noise_type='innovation', **kwargs):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma0:
            raise ValueError(f"Invalid gamma0 value: {gamma0}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, gamma0=gamma0, beta=beta, alpha=alpha,
                        momentum=momentum, noise_type=noise_type)
        super(BelOptimAdaptive, self).__init__(params, defaults)

        self.state['running_avg_grad_sq_sum'] = 0.0
        self.state['running_avg_grad_sum'] = 0.0
        self.state['num_params'] = sum(p.numel() for group in self.param_groups for p in group['params'])


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Calculate global gradient statistics
        grad_sum_sq = 0.0
        grad_sum = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_sum_sq += torch.sum(p.grad.data.pow(2)).item()
                    grad_sum += torch.sum(p.grad.data).item()

        momentum = self.defaults['momentum']
        self.state['running_avg_grad_sq_sum'] = self.state['running_avg_grad_sq_sum'] * momentum + grad_sum_sq * (1 - momentum)
        self.state['running_avg_grad_sum'] = self.state['running_avg_grad_sum'] * momentum + grad_sum * (1 - momentum)

        avg_grad_sq = self.state['running_avg_grad_sq_sum'] / self.state['num_params']
        avg_grad = self.state['running_avg_grad_sum'] / self.state['num_params']

        uncertainty = avg_grad_sq - avg_grad**2
        uncertainty = max(0, uncertainty) # Ensure non-negative

        gamma0 = self.defaults['gamma0']
        alpha = self.defaults['alpha']
        gamma_t = gamma0 * math.exp(-alpha * uncertainty)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                damping = gamma_t * grad.pow(2)
                hamiltonian = group['lr'] * grad

                noise = 0
                if group['beta'] > 0:
                    if group['noise_type'] == 'gaussian':
                        noise = torch.randn_like(grad)
                    elif group['noise_type'] == 'innovation':
                        noise = torch.randn_like(grad)

                stochastic_term = group['beta'] * grad * noise

                p.data.add_(- (damping + hamiltonian) + stochastic_term)

        return loss

class BelOptimLayerwise(Optimizer):
    """
    Belavkin Optimizer with Layerwise Adaptive Parameters.
    Different γ, β parameters per layer based on local gradient statistics.
    """
    def __init__(self, params, lr=1e-3, gamma_base=0.1, beta_base=0.01,
                 noise_type='innovation', **kwargs):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma_base:
            raise ValueError(f"Invalid gamma_base value: {gamma_base}")
        if not 0.0 <= beta_base:
            raise ValueError(f"Invalid beta_base value: {beta_base}")

        defaults = dict(lr=lr, gamma_base=gamma_base, beta_base=beta_base,
                        noise_type=noise_type)
        super(BelOptimLayerwise, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Calculate local gradient statistics for this group
            if not group['params']:
                continue

            grads_in_group = [p.grad.data.flatten() for p in group['params'] if p.grad is not None]
            if not grads_in_group:
                continue

            flat_grad_group = torch.cat(grads_in_group)
            grad_norm = torch.norm(flat_grad_group)

            gamma_base = group['gamma_base']
            beta_base = group['beta_base']

            # Simple scaling based on norm
            gamma = gamma_base * grad_norm
            beta = beta_base * grad_norm

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                damping = gamma * grad.pow(2)
                hamiltonian = group['lr'] * grad

                noise = 0
                if beta > 0:
                    if group['noise_type'] == 'gaussian':
                        noise = torch.randn_like(grad)
                    elif group['noise_type'] == 'innovation':
                        noise = torch.randn_like(grad)

                stochastic_term = beta * grad * noise

                p.data.add_(- (damping + hamiltonian) + stochastic_term)

        return loss
