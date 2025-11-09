import torch
from torch.optim.optimizer import Optimizer

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=1.0, eta=1.0, beta=0.1, ema_decay=0.9):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= eta:
            raise ValueError(f"Invalid eta value: {eta}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"Invalid ema_decay value: {ema_decay}")

        defaults = dict(lr=lr, gamma=gamma, eta=eta, beta=beta, ema_decay=ema_decay)
        super(BelavkinOptimizer, self).__init__(params, defaults)

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
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p.data)

                ema_grad = state['ema_grad']
                ema_decay = group['ema_decay']
                state['step'] += 1

                # Update moving average of gradients
                ema_grad.mul_(ema_decay).add_(grad, alpha=1 - ema_decay)

                # Belavkin update rule
                grad_diff = grad - ema_grad
                adaptive_lr = group['gamma']
                nonlinear_collapse = group['eta']
                stochastic_exploration = group['beta']

                noise = torch.randn_like(grad)

                update = -(adaptive_lr * grad + nonlinear_collapse * grad_diff.pow(2)) + \
                         stochastic_exploration * grad_diff.abs() * noise

                p.data.add_(update, alpha=group['lr'])

        return loss
