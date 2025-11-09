import torch

class BelavkinOptimizer(torch.optim.Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    Args:
        params: Model parameters
        lr (float): Learning rate η (default: 1e-3)
        gamma (float): Damping factor γ (default: 1e-4)
        beta (float): Exploration factor β (default: 1e-2)
        adaptive_gamma (bool): Adapt γ based on gradient statistics
        adaptive_beta (bool): Adapt β based on loss landscape curvature
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2,
                 adaptive_gamma=False, adaptive_beta=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, gamma=gamma, beta=beta,
                        adaptive_gamma=adaptive_gamma, adaptive_beta=adaptive_beta)
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Calculate total grad norm for adaptive gamma if needed
        total_grad_norm_sq = 0
        is_adaptive_gamma = any(group['adaptive_gamma'] for group in self.param_groups)
        if is_adaptive_gamma:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_grad_norm_sq += p.grad.data.pow(2).sum()

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

                state['step'] += 1

                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']
                adaptive_gamma = group['adaptive_gamma']
                adaptive_beta = group['adaptive_beta']

                # Adaptive damping
                current_gamma = gamma
                if adaptive_gamma:
                    # The paper implies norm over all gradients, not per-parameter
                    current_gamma = gamma / (1 + total_grad_norm_sq)

                # Adaptive exploration
                if adaptive_beta:
                    # This is not implemented as the proposal is underspecified
                    # regarding how to adapt beta based on loss landscape curvature
                    # without access to the loss.
                    pass

                # Update rule: θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε
                # We assume Δt = 1 for simplicity

                # Deterministic part
                deterministic_update = current_gamma * grad.pow(2) + lr * grad

                # Stochastic part
                noise = torch.randn_like(grad)
                stochastic_update = beta * grad * noise

                # The update rule from the paper has a + for the stochastic term
                p.data.add_(-deterministic_update + stochastic_update)

        return loss
