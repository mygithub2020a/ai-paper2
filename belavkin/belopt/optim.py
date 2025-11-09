import torch
from torch.optim.optimizer import Optimizer

class BelOpt(Optimizer):
    """
    Implements the Belavkin Optimizer (BelOpt).

    The update rule is given by:
    θ_t+1 = θ_t - (γ * g^2 + η * g) + β * g * ε
    where:
        g = ∇L(θ) (gradient)
        η (lr) is the learning rate.
        γ (gamma0) is the damping coefficient.
        β (beta0) is the stochastic exploration scale.
        ε is a random tensor with elements drawn from N(0, 1).
    """
    def __init__(self, params, lr=1e-3, gamma0=0.0, beta0=0.0, eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma0:
            raise ValueError(f"Invalid gamma0 value: {gamma0}")
        if not 0.0 <= beta0:
            raise ValueError(f"Invalid beta0 value: {beta0}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, gamma0=gamma0, beta0=beta0, eps=eps, weight_decay=weight_decay)
        super(BelOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
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
                    raise RuntimeError('BelOpt does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                # Perform decoupled weight decay
                # AdamW-style weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])

                # Get hyperparameters
                eta = group['lr']
                gamma = group['gamma0']
                beta = group['beta0']

                # Damping term
                damping = grad.pow(2).mul_(gamma)

                # Drift term
                drift = grad.clone().mul_(eta)

                # Combined update
                update = damping.add_(drift)

                # Stochastic exploration term
                if beta > 0:
                    epsilon = torch.randn_like(grad, memory_format=torch.preserve_format)
                    stochastic_term = grad.mul(epsilon).mul_(beta)
                    update.sub_(stochastic_term)

                p.add_(update, alpha=-1)

        return loss
