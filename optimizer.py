"""
Belavkin Optimizer: A novel optimization algorithm derived from quantum filtering theory.

This implementation is based on the Belavkin quantum filtering equation, adapted for
classical optimization with the core update rule:
    dθ = -[γ * (∇L(θ))^2 + η * ∇L(θ)] + β * ∇L(θ) * ε

where:
    γ: adaptive damping factor
    η: learning rate coefficient
    β: stochastic exploration factor
    ε: random noise for exploration
"""

import torch
import torch.optim as optim
from typing import List, Dict, Any, Optional, Callable


class BelavkinOptimizer(optim.Optimizer):
    """
    Implements the Belavkin Optimizer.

    The algorithm combines adaptive second-order information with stochastic exploration,
    motivated by quantum filtering dynamics.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        gamma (float, optional): adaptive damping factor controlling second-order term
            (default: 0.1)
        beta (float, optional): stochastic exploration factor (default: 0.01)
        eps (float, optional): term added to denominator for numerical stability
            (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        momentum (float, optional): momentum coefficient for gradient accumulation
            (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 0.1,
        beta: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "BelavkinOptimizer does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["grad_sq_buffer"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Get hyperparameters
                lr = group["lr"]
                gamma = group["gamma"]
                beta = group["beta"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]

                # Compute the update components
                # 1. Second-order adaptive term: γ * (∇L)^2
                grad_squared = grad * grad
                adaptive_term = gamma * grad_squared

                # 2. First-order gradient term: η * ∇L
                gradient_term = grad

                # 3. Combined deterministic update
                deterministic_update = adaptive_term + gradient_term

                # 4. Stochastic exploration term: β * ∇L * ε
                noise = torch.randn_like(p.data)
                stochastic_term = beta * grad * noise

                # 5. Full update before momentum
                full_update = deterministic_update + stochastic_term

                # Apply weight decay
                if weight_decay != 0:
                    full_update = full_update + weight_decay * p.data

                # Apply momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(full_update, alpha=1)

                # Update parameters
                p.data.add_(buf, alpha=-lr)

        return loss


class AdaptiveBelavkinOptimizer(optim.Optimizer):
    """
    Enhanced Belavkin Optimizer with adaptive hyperparameters.

    This variant adaptively adjusts gamma and beta based on gradient statistics,
    providing improved performance across different problem landscapes.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 0.1,
        beta: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        adaptive_gamma: bool = True,
        adaptive_beta: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            adaptive_gamma=adaptive_gamma,
            adaptive_beta=adaptive_beta,
        )
        super(AdaptiveBelavkinOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step with adaptive parameters."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaptiveBelavkinOptimizer does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["grad_norm_sq"] = 0
                    state["second_moment"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Hyperparameters
                lr = group["lr"]
                gamma = group["gamma"]
                beta = group["beta"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                adaptive_gamma = group["adaptive_gamma"]
                adaptive_beta = group["adaptive_beta"]

                # Compute gradient statistics
                grad_norm_sq = (grad * grad).sum()

                # Update second moment estimate
                state["second_moment"].mul_(0.99).add_(grad * grad, alpha=0.01)
                second_moment = state["second_moment"]

                # Adaptive gamma: scale by inverse of second moment
                if adaptive_gamma:
                    gamma_adaptive = gamma / (torch.sqrt(second_moment) + eps)
                else:
                    gamma_adaptive = gamma

                # Adaptive beta: scale by gradient norm
                if adaptive_beta:
                    grad_norm = torch.sqrt(grad_norm_sq) + eps
                    beta_adaptive = beta / grad_norm
                else:
                    beta_adaptive = beta

                # Compute update components
                grad_squared = grad * grad
                adaptive_term = gamma_adaptive.mean() * grad_squared if adaptive_gamma else gamma * grad_squared
                gradient_term = grad

                deterministic_update = adaptive_term + gradient_term

                # Stochastic exploration
                noise = torch.randn_like(p.data)
                stochastic_term = beta_adaptive * grad * noise if adaptive_beta else beta * grad * noise

                # Full update
                full_update = deterministic_update + stochastic_term

                # Weight decay
                if weight_decay != 0:
                    full_update = full_update + weight_decay * p.data

                # Momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(full_update, alpha=1)

                # Update parameters
                p.data.add_(buf, alpha=-lr)

        return loss


def create_optimizer(
    params,
    optimizer_name: str = "belavkin",
    **kwargs
) -> optim.Optimizer:
    """
    Factory function to create various optimizers.

    Args:
        params: Model parameters
        optimizer_name: Name of optimizer ('belavkin', 'adaptive_belavkin', 'adam', 'sgd', 'rmsprop')
        **kwargs: Additional optimizer hyperparameters

    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "belavkin":
        return BelavkinOptimizer(params, **kwargs)
    elif optimizer_name == "adaptive_belavkin":
        return AdaptiveBelavkinOptimizer(params, **kwargs)
    elif optimizer_name == "adam":
        return optim.Adam(params, **kwargs)
    elif optimizer_name == "sgd":
        return optim.SGD(params, **kwargs)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
