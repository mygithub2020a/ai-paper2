import torch
from torch.optim.optimizer import Optimizer

class BelOptim(Optimizer):
    """
    Belavkin Optimizer derived from quantum filtering equation.

    Args:
        params: Model parameters
        lr (η): Base learning rate
        gamma (γ): Adaptive damping factor
        beta (β): Stochastic exploration factor
        noise_type: 'gaussian', 'innovation', or 'none'
    """
    def __init__(self, params, lr=1e-3, gamma=0.1, beta=0.01,
                 noise_type='innovation', **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, gamma=gamma, beta=beta, noise_type=noise_type)
        super(BelOptim, self).__init__(params, defaults)

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

                state['step'] += 1

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']
                noise_type = group['noise_type']

                # Core update rule: dθ = -[γ * (∇L(θ))² + η * ∇L(θ)] + β * ∇L(θ) * ε
                damping = gamma * grad.pow(2)
                hamiltonian = lr * grad

                noise = 0
                if beta > 0:
                    if noise_type == 'gaussian':
                        noise = torch.randn_like(grad)
                    elif noise_type == 'innovation':
                        # Simplified innovation process for now
                        noise = torch.randn_like(grad)

                stochastic_term = beta * grad * noise

                p.data.add_(- (damping + hamiltonian) + stochastic_term)

        return loss
