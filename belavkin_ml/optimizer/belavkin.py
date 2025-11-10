"""
Belavkin Optimizer - Core Implementation

Implements the quantum filtering-inspired update rule:
    θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

where ε ~ N(0, 1) is standard Gaussian noise.

Theoretical Motivation:
- Damping term γ(∇L)²: Represents measurement backaction
- Drift term η∇L: Standard gradient descent component
- Stochastic term β∇L*ε: State-dependent diffusion (multiplicative noise)
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Callable
import math


class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    This optimizer implements a stochastic update rule derived from the
    Belavkin quantum filtering equation, adapted for classical neural network
    optimization.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate η (default: 1e-3)
        gamma (float, optional): Damping factor γ (default: 1e-4)
        beta (float, optional): Exploration factor β (default: 1e-2)
        dt (float, optional): Time step Δt (default: 1.0)
        adaptive_gamma (bool, optional): Adapt γ based on gradient statistics
            (default: False)
        adaptive_beta (bool, optional): Adapt β based on loss landscape curvature
            (default: False)
        gamma_decay (float, optional): Decay rate for adaptive gamma (default: 0.99)
        beta_decay (float, optional): Decay rate for adaptive beta (default: 0.99)
        grad_clip (float, optional): Gradient clipping threshold (default: None)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): Use AMSGrad-style running statistics (default: False)

    Example:
        >>> optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, gamma=1e-4)
        >>> optimizer.zero_grad()
        >>> loss = loss_fn(model(input), target)
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        dt: float = 1.0,
        adaptive_gamma: bool = False,
        adaptive_beta: bool = False,
        gamma_decay: float = 0.99,
        beta_decay: float = 0.99,
        grad_clip: Optional[float] = None,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= dt:
            raise ValueError(f"Invalid dt value: {dt}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma_decay <= 1.0:
            raise ValueError(f"Invalid gamma_decay value: {gamma_decay}")
        if not 0.0 <= beta_decay <= 1.0:
            raise ValueError(f"Invalid beta_decay value: {beta_decay}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            dt=dt,
            adaptive_gamma=adaptive_gamma,
            adaptive_beta=adaptive_beta,
            gamma_decay=gamma_decay,
            beta_decay=beta_decay,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelavkinOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Implements the update rule:
            θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float, optional): Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            dt = group['dt']
            adaptive_gamma = group['adaptive_gamma']
            adaptive_beta = group['adaptive_beta']
            gamma_decay = group['gamma_decay']
            beta_decay = group['beta_decay']
            grad_clip = group['grad_clip']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Apply gradient clipping if specified
                if grad_clip is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / (grad_norm + 1e-8))

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Running average of gradient statistics
                    state['grad_sq_avg'] = torch.zeros_like(p)
                    # For AMSGrad
                    if amsgrad:
                        state['max_grad_sq_avg'] = torch.zeros_like(p)
                    # Adaptive parameters
                    state['gamma_t'] = gamma
                    state['beta_t'] = beta

                state['step'] += 1
                grad_sq_avg = state['grad_sq_avg']

                # Update running average of squared gradients
                grad_sq_avg.mul_(gamma_decay).addcmul_(grad, grad, value=1 - gamma_decay)

                # Adaptive gamma: γ_t = γ_0 / (1 + ||∇L||²)^α
                if adaptive_gamma:
                    grad_norm_sq = torch.sum(grad * grad)
                    alpha = 0.5  # Tunable parameter
                    state['gamma_t'] = gamma / (1 + grad_norm_sq) ** alpha
                else:
                    state['gamma_t'] = gamma

                # Adaptive beta: Adjust based on gradient variance
                if adaptive_beta:
                    # Use running average of squared gradients as proxy for variance
                    grad_variance = torch.mean(grad_sq_avg)
                    state['beta_t'] = beta * torch.sqrt(grad_variance + 1e-8).item()
                else:
                    state['beta_t'] = beta

                gamma_t = state['gamma_t']
                beta_t = state['beta_t']

                # Compute gradient squared term (element-wise)
                grad_squared = grad * grad

                # Deterministic update: -[γ*(∇L)² + η*∇L]Δt
                # Note: γ term is quadratic damping, η is learning rate
                deterministic_update = -(gamma_t * grad_squared + lr * grad) * dt

                # Stochastic update: β*∇L*√Δt*ε where ε ~ N(0,1)
                # Multiplicative noise scaled by gradient magnitude
                noise = torch.randn_like(grad)
                stochastic_update = beta_t * grad * math.sqrt(dt) * noise

                # Combined update
                update = deterministic_update + stochastic_update

                # Apply update to parameters
                p.add_(update)

        return loss


class BelavkinOptimizerNaturalGradient(BelavkinOptimizer):
    """
    Belavkin optimizer with natural gradient variant using Fisher information.

    This extends the base Belavkin optimizer by incorporating Fisher information
    matrix preconditioning, similar to natural gradient descent.

    Args:
        params: Model parameters
        lr: Learning rate
        gamma: Damping factor
        beta: Exploration factor
        fisher_decay: Decay rate for Fisher information estimation (default: 0.99)
        fisher_eps: Regularization for Fisher matrix (default: 1e-4)
        **kwargs: Additional arguments passed to BelavkinOptimizer
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        fisher_decay: float = 0.99,
        fisher_eps: float = 1e-4,
        **kwargs
    ):
        super().__init__(params, lr=lr, gamma=gamma, beta=beta, **kwargs)
        self.fisher_decay = fisher_decay
        self.fisher_eps = fisher_eps

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Step with Fisher information preconditioning.

        Approximates natural gradient by using running average of squared
        gradients as diagonal Fisher information matrix.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            dt = group['dt']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['fisher'] = torch.zeros_like(p)

                state['step'] += 1
                fisher = state['fisher']

                # Update Fisher information estimate (diagonal approximation)
                fisher.mul_(self.fisher_decay).addcmul_(
                    grad, grad, value=1 - self.fisher_decay
                )

                # Preconditioned gradient: F^{-1} * grad
                # Using diagonal approximation: grad / (fisher + eps)
                precond_grad = grad / (fisher.sqrt() + self.fisher_eps)

                # Apply Belavkin update with preconditioned gradient
                grad_squared = precond_grad * precond_grad

                deterministic_update = -(gamma * grad_squared + lr * precond_grad) * dt

                noise = torch.randn_like(grad)
                stochastic_update = beta * precond_grad * math.sqrt(dt) * noise

                update = deterministic_update + stochastic_update
                p.add_(update)

        return loss
