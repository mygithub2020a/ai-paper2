import torch
from torch.optim.optimizer import Optimizer
from .schedules import ConstantScheduler

class BelOpt(Optimizer):
    def __init__(self, params, eta_scheduler=None, gamma_scheduler=None, beta_scheduler=None, eps=1e-8,
                 decoupled_weight_decay=0.0, update_clip=None):

        if eta_scheduler is None:
            eta_scheduler = ConstantScheduler(1e-3)
        if gamma_scheduler is None:
            gamma_scheduler = ConstantScheduler(1e-3)
        if beta_scheduler is None:
            beta_scheduler = ConstantScheduler(1e-3)

        defaults = dict(eta_scheduler=eta_scheduler, gamma_scheduler=gamma_scheduler, beta_scheduler=beta_scheduler,
                        eps=eps, decoupled_weight_decay=decoupled_weight_decay,
                        update_clip=update_clip)
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
                    # EMA of grad^2
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Get scheduled values
                eta_t = group['eta_scheduler'](state['step'])
                gamma0 = group['gamma_scheduler'](state['step'])
                beta_t = group['beta_scheduler'](state['step'])

                # Decoupled weight decay
                if group['decoupled_weight_decay'] > 0:
                    p.mul_(1 - eta_t * group['decoupled_weight_decay'])

                # Belavkin update
                eps = group['eps']

                # Adaptive damping
                exp_avg_sq.mul_(0.9).addcmul_(grad, grad, value=0.1)
                vt = exp_avg_sq
                gamma_t = gamma0 / (torch.sqrt(vt) + eps)

                # Update rule
                gt = grad
                term1 = gamma_t * gt * gt
                term2 = eta_t * gt
                term3 = beta_t * gt * torch.randn_like(gt)

                update = term1 + term2 - term3

                # Update clipping
                if group['update_clip'] is not None:
                    update_norm = torch.norm(update)
                    if update_norm > group['update_clip']:
                        update.mul_(group['update_clip'] / update_norm)

                p.add_(-update)

        return loss
