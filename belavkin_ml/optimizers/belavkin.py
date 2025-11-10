"""
Belavkin-inspired optimizer for PyTorch.

Implements optimization algorithm derived from quantum filtering principles
in the Belavkin equation framework.

Update rule:
    θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

where:
    - θ: Network parameters (analogue of quantum state)
    - ∇L(θ): Loss gradient (analogue of measurement signal)
    - γ: Adaptive damping factor (analogue of measurement strength)
    - η: Learning rate (drift coefficient)
    - β: Stochastic exploration factor (diffusion coefficient)
    - ε: Gaussian noise term (measurement uncertainty)
"""

import math
from typing import List, Optional, Callable, Tuple
import torch
from torch.optim.optimizer import Optimizer


class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    This optimizer implements a novel update rule inspired by the Belavkin
    quantum filtering equation, featuring:
    1. Quadratic damping term proportional to gradient magnitude squared
    2. Gradient-dependent multiplicative noise for exploration
    3. Adaptive parameter tuning based on gradient statistics

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate η (default: 1e-3)
        gamma (float, optional): Base damping factor γ₀ (default: 1e-4)
        beta (float, optional): Base exploration factor β₀ (default: 1e-2)
        adaptive_gamma (bool, optional): Enable adaptive damping (default: True)
        adaptive_beta (bool, optional): Enable adaptive exploration (default: False)
        gamma_decay (float, optional): Decay exponent α for adaptive γ (default: 0.5)
        beta_decay (float, optional): Decay for adaptive β (default: 0.5)
        grad_clip (float, optional): Gradient clipping threshold (default: None)
        eps (float, optional): Small constant for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): Use AMSGrad-style max tracking (default: False)

    Example:
        >>> optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3)
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
        adaptive_gamma: bool = True,
        adaptive_beta: bool = False,
        gamma_decay: float = 0.5,
        beta_decay: float = 0.5,
        grad_clip: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            adaptive_gamma=adaptive_gamma,
            adaptive_beta=adaptive_beta,
            gamma_decay=gamma_decay,
            beta_decay=beta_decay,
            grad_clip=grad_clip,
            eps=eps,
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

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float, optional): Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma_base = group['gamma']
            beta_base = group['beta']
            adaptive_gamma = group['adaptive_gamma']
            adaptive_beta = group['adaptive_beta']
            gamma_decay = group['gamma_decay']
            beta_decay = group['beta_decay']
            grad_clip = group['grad_clip']
            eps = group['eps']
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
                    grad = torch.clamp(grad, -grad_clip, grad_clip)

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Running statistics for adaptive parameters
                    param_state['grad_sq_sum'] = torch.zeros_like(p)
                    param_state['grad_sum'] = torch.zeros_like(p)
                    if amsgrad:
                        param_state['max_grad_sq'] = torch.zeros_like(p)

                grad_sq_sum = param_state['grad_sq_sum']
                grad_sum = param_state['grad_sum']
                param_state['step'] += 1
                step = param_state['step']

                # Compute gradient squared
                grad_sq = grad.pow(2)

                # Update running statistics (exponential moving average)
                beta1 = 0.9  # Momentum for gradient statistics
                grad_sq_sum.mul_(beta1).add_(grad_sq, alpha=1 - beta1)
                grad_sum.mul_(beta1).add_(grad, alpha=1 - beta1)

                # AMSGrad: track maximum gradient squared
                if amsgrad:
                    max_grad_sq = param_state['max_grad_sq']
                    torch.max(max_grad_sq, grad_sq_sum, out=max_grad_sq)
                    grad_sq_eff = max_grad_sq
                else:
                    grad_sq_eff = grad_sq_sum

                # Compute adaptive damping factor
                if adaptive_gamma:
                    # γ_t = γ₀ * (1 + ||∇L||²)^(-α)
                    grad_norm_sq = grad_sq.sum()
                    gamma = gamma_base / (1 + grad_norm_sq).pow(gamma_decay)
                else:
                    gamma = gamma_base

                # Compute adaptive exploration factor
                if adaptive_beta:
                    # β_t based on local curvature estimate
                    # Use ratio of gradient variance to mean
                    grad_var = grad_sq_eff - grad_sum.pow(2) + eps
                    beta = beta_base * torch.sqrt(grad_var + eps).mean()
                else:
                    beta = beta_base

                # Belavkin update rule:
                # θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

                # Damping term: γ * (∇L)²
                damping = gamma * grad_sq_eff

                # Drift term: η * ∇L
                drift = lr * grad

                # Stochastic term: β * ∇L * ε where ε ~ N(0, 1)
                # Multiplicative noise scaled by gradient magnitude
                noise = torch.randn_like(grad)
                stochastic = beta * torch.abs(grad) * noise

                # Combined update with proper time scaling
                # Δt = 1 (discrete time step)
                # √Δt = 1 for discrete case
                update = damping + drift - stochastic

                # Apply update
                p.add_(-update)

        return loss


class BelavkinSGD(BelavkinOptimizer):
    """
    Belavkin optimizer with SGD-like defaults (minimal stochasticity).

    This variant emphasizes the deterministic damping component with
    minimal exploration noise, similar to SGD with momentum.

    Args:
        params: Model parameters
        lr (float): Learning rate (default: 1e-2)
        gamma (float): Damping factor (default: 1e-3)
        beta (float): Exploration factor (default: 1e-4)
        momentum (float): Momentum factor for gradient statistics (default: 0.9)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        gamma: float = 1e-3,
        beta: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0,
        **kwargs
    ):
        super().__init__(
            params,
            lr=lr,
            gamma=gamma,
            beta=beta,
            adaptive_gamma=True,
            adaptive_beta=False,
            weight_decay=weight_decay,
            **kwargs
        )


class BelavkinAdam(BelavkinOptimizer):
    """
    Belavkin optimizer with Adam-like adaptive behavior.

    This variant uses adaptive damping and exploration, with parameters
    tuned to behave similarly to Adam but with quantum filtering principles.

    Args:
        params: Model parameters
        lr (float): Learning rate (default: 1e-3)
        gamma (float): Base damping factor (default: 1e-4)
        beta (float): Base exploration factor (default: 1e-2)
        amsgrad (bool): Use AMSGrad variant (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        amsgrad: bool = True,
        weight_decay: float = 0,
        **kwargs
    ):
        super().__init__(
            params,
            lr=lr,
            gamma=gamma,
            beta=beta,
            adaptive_gamma=True,
            adaptive_beta=True,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            **kwargs
        )


def get_belavkin_optimizer(
    params,
    variant: str = "default",
    **kwargs
) -> BelavkinOptimizer:
    """
    Factory function to create Belavkin optimizer variants.

    Args:
        params: Model parameters
        variant (str): Optimizer variant ('default', 'sgd', 'adam')
        **kwargs: Additional optimizer arguments

    Returns:
        BelavkinOptimizer instance
    """
    variants = {
        'default': BelavkinOptimizer,
        'sgd': BelavkinSGD,
        'adam': BelavkinAdam,
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    return variants[variant](params, **kwargs)
