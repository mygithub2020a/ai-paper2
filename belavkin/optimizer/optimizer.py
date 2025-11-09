import torch
from torch.optim.optimizer import Optimizer, required

class BelavkinOptimizer(Optimizer):
    def __init__(self, params, lr=required, gamma=0.1, eta=0.1, beta=0.01, ema_decay=0.9):
        if lr is not required and lr < 0.0:
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

    def __setstate__(self, state):
        super(BelavkinOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
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
                    raise RuntimeError('BelavkinOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['ema_grad'] = torch.zeros_like(grad, memory_format=torch.preserve_format)

                ema_grad = state['ema_grad']
                ema_decay = group['ema_decay']

                state['step'] += 1

                # Update EMA of gradients
                ema_grad.mul_(ema_decay).add_(grad, alpha=1 - ema_decay)

                m = ema_grad

                # Update rule
                d_theta = -(group['gamma'] * grad + group['eta'] * (grad - m)**2) + \
                          group['beta'] * torch.abs(grad - m) * torch.randn_like(grad)

                p.add_(d_theta, alpha=group['lr'])

        return loss
