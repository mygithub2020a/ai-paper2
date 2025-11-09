import torch
from torch.optim.optimizer import Optimizer

class BelOpt(Optimizer):
    def __init__(self, params, lr=1e-3, gamma0=0.0, beta0=0.0, eps=1e-8,
                 decoupled_weight_decay=0.0, grad_clip=None, update_clip=None,
                 adaptive_gamma=False, adaptive_beta=False, ema_alpha=0.9):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma0:
            raise ValueError(f"Invalid gamma0 value: {gamma0}")
        if not 0.0 <= beta0:
            raise ValueError(f"Invalid beta0 value: {beta0}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, gamma0=gamma0, beta0=beta0, eps=eps,
                        decoupled_weight_decay=decoupled_weight_decay,
                        grad_clip=grad_clip, update_clip=update_clip,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta,
                        ema_alpha=ema_alpha)
        super(BelOpt, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelOpt, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

            if len(params_with_grad) == 0:
                continue

            # Gradient clipping
            if group['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(params_with_grad, group['grad_clip'])

            for p in params_with_grad:
                if p.grad.is_sparse:
                    raise RuntimeError("BelOpt does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if group['adaptive_gamma'] or group['adaptive_beta']:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                # Decoupled weight decay
                if group['decoupled_weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['decoupled_weight_decay'])

                # Core update calculation
                grad = p.grad

                gamma = group['gamma0']
                if group['adaptive_gamma']:
                    exp_avg_sq = state['exp_avg_sq']
                    alpha = group['ema_alpha']
                    exp_avg_sq.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                    gamma = group['gamma0'] / (exp_avg_sq.sqrt() + group['eps'])

                beta = group['beta0']
                if group['adaptive_beta']:
                    # Can reuse the same EMA for both
                    if not group['adaptive_gamma']:
                        exp_avg_sq = state['exp_avg_sq']
                        alpha = group['ema_alpha']
                        exp_avg_sq.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                    beta = group['beta0'] / (exp_avg_sq.sqrt() + group['eps'])

                eta = group['lr']

                # Descent part: -(γ*g^2 + η*g)
                descent_part = gamma * grad.pow(2) + eta * grad

                # Total update starts with descent
                update = -descent_part

                # Exploration part: +β*g*ε
                if beta > 0:
                    noise = torch.randn_like(grad)
                    update.add_(beta * grad * noise)

                # Update clipping
                if group['update_clip'] is not None:
                    update.clamp_(-group['update_clip'], group['update_clip'])

                # Apply the final update
                p.add_(update)

        return loss
