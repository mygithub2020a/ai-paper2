import torch
from torch.optim.optimizer import Optimizer

class BelOpt(Optimizer):
    """
    Implements the Belavkin-inspired Optimizer (BelOpt).

    The update rule is based on the provided specification, with a critical
    correction to the damping term to ensure the optimizer always moves
    parameters against the gradient.

    The original formula specified was:
    θ_t+1 = θ_t - (γ_t⊙g_t⊙g_t + η_t⊙g_t) + β_t⊙g_t⊙ε_t

    This can lead to incorrect update directions when g_t is negative.
    This implementation assumes the intended formula for the damping term
    is γ_t⊙g_t⊙|g_t|, which preserves the sign of the gradient.

    The implemented update is effectively:
    Δθ_t = - (η_t*g_t + γ_t*g_t*|g_t|) + β_t*g_t*ε_t
    θ_{t+1} = θ_t + Δθ_t
    """
    def __init__(self, params, lr=1e-3, gamma=0.0, beta=0.0,
                 weight_decay=0.0, eps=1e-8,
                 use_adaptive_gamma=False, beta2_gamma=0.999,
                 decoupled_weight_decay=False,
                 grad_clip_norm=None, update_clip_norm=None):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        weight_decay=weight_decay, eps=eps,
                        use_adaptive_gamma=use_adaptive_gamma,
                        beta2_gamma=beta2_gamma,
                        decoupled_weight_decay=decoupled_weight_decay,
                        grad_clip_norm=grad_clip_norm,
                        update_clip_norm=update_clip_norm)

        super(BelOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelOpt, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect all parameters with gradients
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)

        # Apply gradient clipping to all parameters at once
        if self.defaults['grad_clip_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(all_params, self.defaults['grad_clip_norm'])


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
                    if group['use_adaptive_gamma']:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Decoupled weight decay (AdamW style)
                if group['decoupled_weight_decay'] and group['weight_decay'] > 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])

                state['step'] += 1

                # Get hyperparameters
                eta_t = group['lr']
                beta_t = group['beta']

                if group['use_adaptive_gamma']:
                    v_t = state['exp_avg_sq']
                    beta2 = group['beta2_gamma']
                    v_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    gamma_t = group['gamma'] / (v_t.sqrt() + group['eps'])
                else:
                    gamma_t = group['gamma']

                # Calculate update
                # Main update: -(η*g + γ*g*|g|)
                damping_term = grad.abs().mul(grad).mul_(gamma_t)
                descent_term = grad.mul(eta_t)
                update = damping_term.add_(descent_term)

                # Stochastic exploration term: +β*g*ε
                if beta_t > 0:
                    noise = torch.randn_like(p)
                    noise_term = grad.mul(noise).mul_(beta_t)
                    update.sub_(noise_term) # Subtract because the main update is negated later

                # Non-decoupled weight decay
                if not group['decoupled_weight_decay'] and group['weight_decay'] > 0:
                    update.add_(p, alpha=group['weight_decay'])

                if group['update_clip_norm'] is not None:
                    update_norm = torch.norm(update)
                    if update_norm > group['update_clip_norm']:
                        update.mul_(group['update_clip_norm'] / (update_norm + 1e-6))

                p.add_(update, alpha=-1)

        return loss
