"""
Belavkin Optimizer Implementation

Core update rule derived from quantum stochastic filtering:
dθ = -[γ * (∇L(θ))^2 + η * ∇L(θ)] + β * ∇L(θ) * ε

where:
- γ: adaptive damping factor (controls curvature-based adaptation)
- η: base learning rate (similar to standard gradient descent)
- β: stochastic exploration factor (quantum noise amplitude)
- ε: random noise term (typically Gaussian)

References:
- Belavkin, V.P. "Quantum Stochastic Calculus and Quantum Nonlinear Filtering"
- https://arxiv.org/abs/math/0512510
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import math


class BelavkinOptimizer(Optimizer):
    """
    Implements Belavkin optimizer derived from quantum filtering equations.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (η parameter) (default: 1e-3)
        gamma (float, optional): adaptive damping factor (γ parameter) (default: 1e-4)
        beta (float, optional): stochastic exploration factor (β parameter) (default: 1e-5)
        noise_type (str, optional): type of noise for exploration ('gaussian' or 'uniform')
            (default: 'gaussian')
        momentum (float, optional): momentum factor (default: 0.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        adaptive_gamma (bool, optional): whether to adapt gamma based on gradient history
            (default: True)
        eps (float, optional): term added to denominator for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = BelavkinOptimizer(model.parameters(), lr=0.001, gamma=1e-4, beta=1e-5)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-5,
        noise_type: str = 'gaussian',
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        adaptive_gamma: bool = True,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if noise_type not in ['gaussian', 'uniform']:
            raise ValueError(f"Invalid noise_type: {noise_type}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            noise_type=noise_type,
            momentum=momentum,
            weight_decay=weight_decay,
            adaptive_gamma=adaptive_gamma,
            eps=eps,
        )
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelavkinOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            noise_type = group['noise_type']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            adaptive_gamma = group['adaptive_gamma']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    if adaptive_gamma:
                        param_state['grad_sq_ema'] = torch.zeros_like(p.data)
                        param_state['ema_decay'] = 0.9

                param_state['step'] += 1

                # Adaptive gamma based on gradient history
                if adaptive_gamma:
                    ema_decay = param_state['ema_decay']
                    grad_sq_ema = param_state['grad_sq_ema']

                    # Update exponential moving average of squared gradients
                    grad_sq_ema.mul_(ema_decay).addcmul_(grad, grad, value=1 - ema_decay)
                    param_state['grad_sq_ema'] = grad_sq_ema

                    # Adapt gamma based on gradient magnitude
                    adaptive_gamma_factor = 1.0 / (torch.sqrt(grad_sq_ema) + eps)
                    effective_gamma = gamma * adaptive_gamma_factor
                else:
                    effective_gamma = gamma

                # Generate noise term ε
                if noise_type == 'gaussian':
                    noise = torch.randn_like(grad)
                elif noise_type == 'uniform':
                    noise = torch.rand_like(grad) * 2 - 1  # Uniform in [-1, 1]
                else:
                    noise = torch.zeros_like(grad)

                # Compute Belavkin update rule:
                # dθ = -[γ * (∇L(θ))^2 + η * ∇L(θ)] + β * ∇L(θ) * ε

                # Damping term: γ * (∇L(θ))^2 (element-wise)
                if adaptive_gamma:
                    damping_term = effective_gamma * grad
                else:
                    damping_term = effective_gamma * grad.pow(2).sign() * grad

                # Base gradient term: η * ∇L(θ)
                base_term = lr * grad

                # Stochastic exploration term: β * ∇L(θ) * ε
                exploration_term = beta * grad * noise

                # Combine all terms
                update = -(damping_term + base_term) + exploration_term

                # Apply momentum if specified
                if momentum != 0:
                    momentum_buffer = param_state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(update)
                    update = momentum_buffer

                # Apply update to parameters
                p.data.add_(update)

        return loss


class BelavkinOptimizerV2(Optimizer):
    """
    Alternative formulation of Belavkin optimizer with different damping interpretation.

    This version treats the damping term as a quadratic penalty on gradient magnitude,
    providing stronger regularization on large gradients.

    Update rule:
    dθ = -η * ∇L(θ) - γ * ||∇L(θ)||^2 * ∇L(θ) + β * ∇L(θ) * ε
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-6,
        beta: float = 1e-5,
        noise_type: str = 'gaussian',
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            noise_type=noise_type,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
        )
        super(BelavkinOptimizerV2, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            noise_type = group['noise_type']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                param_state['step'] += 1

                # Generate noise
                if noise_type == 'gaussian':
                    noise = torch.randn_like(grad)
                else:
                    noise = torch.rand_like(grad) * 2 - 1

                # Compute gradient magnitude squared
                grad_mag_sq = (grad.pow(2).sum() + eps).sqrt()

                # Update: dθ = -η * ∇L - γ * ||∇L||^2 * ∇L + β * ∇L * ε
                base_term = lr * grad
                damping_term = gamma * grad_mag_sq * grad
                exploration_term = beta * grad * noise

                update = -(base_term + damping_term) + exploration_term

                if momentum != 0:
                    momentum_buffer = param_state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(update)
                    update = momentum_buffer

                p.data.add_(update)

        return loss
