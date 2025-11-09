import torch
from torch.optim.optimizer import Optimizer

class Belavkin(Optimizer):
    """
    Implements the Belavkin Optimizer, derived from the Belavkin quantum filtering equation.
    """

    def __init__(self, params, lr=1e-3, eta=1.0, beta=0.1, gamma_init=0.1,
                 rho=0.99, eps=1e-8, adaptive_gamma=True, grad_clip=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eta:
            raise ValueError(f"Invalid eta value: {eta}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= gamma_init:
            raise ValueError(f"Invalid gamma_init value: {gamma_init}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, eta=eta, beta=beta, gamma_init=gamma_init,
                        rho=rho, eps=eps, adaptive_gamma=adaptive_gamma,
                        grad_clip=grad_clip)
        super().__init__(params, defaults)

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
                g = p.grad
                if group['grad_clip'] is not None:
                    g.clamp_(-group['grad_clip'], group['grad_clip'])
                state = self.state[p]
                if 'm2' not in state:
                    state['m2'] = torch.zeros_like(p)
                m2 = state['m2']
                m2.mul_(group['rho']).addcmul_(g, g, value=1-group['rho'])
                gamma = 1.0 / (torch.sqrt(m2) + group['eps']) if group['adaptive_gamma'] else group['gamma_init']
                noise = torch.randn_like(g)

                term_damping = g.mul(g).mul(gamma)
                term_damping_drift = g.mul(group['eta']).add(term_damping)
                term_noise = g.mul(noise).mul(group['beta'])
                delta = term_noise.sub_(term_damping_drift)

                p.add_(delta, alpha=-group['lr'])
        return loss
