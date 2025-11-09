import torch
from torch.optim.optimizer import Optimizer

class Belavkin(Optimizer):
    """Implements the Belavkin optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (alpha_t) (default: 1e-3)
        eta (float, optional): drift term coefficient (eta_t) (default: 1.0)
        beta (float, optional): stochastic exploration term coefficient (beta_t) (default: 0.1)
        gamma_init (float, optional): initial value for the adaptive damping term (gamma_t) (default: 0.1)
        rho (float, optional): coefficient for the moving average of squared gradients (default: 0.99)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        adaptive_gamma (bool, optional): whether to use the adaptive damping term (default: True)
        grad_clip (float, optional): max norm of the gradients (default: None)
        grad_scaled_noise (bool, optional): whether to use gradient-scaled noise (default: False)
        eps0 (float, optional): stability term for gradient-scaled noise (default: 1e-6)
    """
    def __init__(self, params, lr=1e-3, eta=1.0, beta=0.1, gamma_init=0.1,
                 rho=0.99, eps=1e-8, adaptive_gamma=True, grad_clip=None,
                 grad_scaled_noise=False, eps0=1e-6):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eta:
            raise ValueError(f"Invalid eta value: {eta}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= gamma_init:
            raise ValueError(f"Invalid gamma_init value: {gamma_init}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError(f"Invalid grad_clip value: {grad_clip}")

        defaults = dict(lr=lr, eta=eta, beta=beta, gamma_init=gamma_init,
                        rho=rho, eps=eps, adaptive_gamma=adaptive_gamma,
                        grad_clip=grad_clip, grad_scaled_noise=grad_scaled_noise,
                        eps0=eps0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

            if not params_with_grad:
                continue

            if group['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(params_with_grad, group['grad_clip'])

            lr = group['lr']
            eta = group['eta']
            beta = group['beta']
            rho = group['rho']
            eps = group['eps']
            adaptive_gamma = group['adaptive_gamma']
            grad_scaled_noise = group['grad_scaled_noise']
            eps0 = group['eps0']

            for p in params_with_grad:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # State initialization
                if 'm2' not in state:
                    state['m2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m2 = state['m2']
                m2.mul_(rho).addcmul_(g, g, value=1 - rho)

                gamma = group['gamma_init']
                if adaptive_gamma:
                    gamma = 1.0 / (m2.sqrt().add_(eps))

                noise = torch.randn_like(g, memory_format=torch.preserve_format)
                if grad_scaled_noise:
                    noise.div_(g.abs().add_(eps0))

                # Canonical PyTorch implementation:
                # Final Corrected Update Rule for Gradient Descent:
                # Final Corrected Update Rule for Gradient Descent:
                # The update rule from the brief is: Δθ_t = -(γ*g^2 + η*g) + β*(g*ε)
                # The parameter update is: θ_{t+1} = θ_t - lr * Δθ_t
                # PyTorch's p.data.add_ implements: p.data = p.data + alpha * delta
                # To match the formula, we set delta = Δθ_t and alpha = -lr
                delta = -(gamma * g.pow(2) + eta * g) + beta * (g * noise)

                p.data.add_(delta, alpha=-lr)

        return loss
