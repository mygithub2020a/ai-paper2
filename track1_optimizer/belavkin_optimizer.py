"""
Belavkin-Inspired Optimizer for Neural Networks

This module implements a novel optimization algorithm inspired by Belavkin's
quantum filtering equations. The optimizer incorporates quantum filtering
principles including measurement backaction, state-dependent diffusion, and
adaptive damping.

Theoretical Background:
    The Belavkin equation for quantum filtering:
        dψ_t = -[(1/2)L*L + (i/ℏ)H]ψ_t dt + Lψ_t dy_t

    Adapted to neural network optimization:
        dθ = -[γ * (∇L(θ))² + η * ∇L(θ)] dt + β * ∇L(θ) * dε_t

    Where:
        - θ: Network parameters (analogue of quantum state)
        - ∇L(θ): Loss gradient (analogue of measurement signal)
        - γ: Damping factor (analogue of measurement strength)
        - η: Learning rate (drift coefficient)
        - β: Exploration factor (diffusion coefficient)
        - ε_t: Gaussian noise (measurement uncertainty)

References:
    - Belavkin, V.P. (1992). Quantum stochastic calculus and quantum nonlinear filtering
    - Belavkin, V.P. (2005). arXiv:math/0512510
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Callable
import math


class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    The optimizer implements the update rule:
        θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

    This combines:
        1. Standard gradient descent (η*∇L)
        2. Adaptive damping based on gradient magnitude (γ*(∇L)²)
        3. Gradient-dependent stochastic exploration (β*∇L*ε)

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Learning rate η (default: 1e-3)
        gamma (float, optional): Damping factor γ (default: 1e-4)
        beta (float, optional): Exploration factor β (default: 1e-2)
        adaptive_gamma (bool, optional): Adapt γ based on gradient statistics
            (default: False)
        adaptive_beta (bool, optional): Adapt β based on loss landscape curvature
            (default: False)
        gamma_decay (float, optional): Decay rate for adaptive gamma (default: 0.999)
        beta_decay (float, optional): Decay rate for adaptive beta (default: 0.999)
        clip_value (float, optional): Gradient clipping value (default: 10.0)
        natural_gradient (bool, optional): Use Fisher information approximation
            (default: False)
        dt (float, optional): Time step size (default: 1.0)
        eps (float, optional): Term added for numerical stability (default: 1e-8)

    Example:
        >>> optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, gamma=1e-4)
        >>> for input, target in dataset:
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(input), target)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        adaptive_gamma: bool = False,
        adaptive_beta: bool = False,
        gamma_decay: float = 0.999,
        beta_decay: float = 0.999,
        clip_value: float = 10.0,
        natural_gradient: bool = False,
        dt: float = 1.0,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            adaptive_gamma=adaptive_gamma,
            adaptive_beta=adaptive_beta,
            gamma_decay=gamma_decay,
            beta_decay=beta_decay,
            clip_value=clip_value,
            natural_gradient=natural_gradient,
            dt=dt,
            eps=eps,
        )
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelavkinOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('adaptive_gamma', False)
            group.setdefault('adaptive_beta', False)
            group.setdefault('natural_gradient', False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float, optional): The loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            adaptive_gamma = group['adaptive_gamma']
            adaptive_beta = group['adaptive_beta']
            gamma_decay = group['gamma_decay']
            beta_decay = group['beta_decay']
            clip_value = group['clip_value']
            natural_gradient = group['natural_gradient']
            dt = group['dt']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Gradient clipping for numerical stability
                if clip_value is not None:
                    grad = torch.clamp(grad, -clip_value, clip_value)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_sq_avg'] = torch.zeros_like(p)
                    state['grad_avg'] = torch.zeros_like(p)
                    if natural_gradient:
                        state['fisher_approx'] = torch.ones_like(p)

                state['step'] += 1
                grad_sq_avg = state['grad_sq_avg']
                grad_avg = state['grad_avg']

                # Update running averages for adaptive parameters
                grad_sq_avg.mul_(gamma_decay).add_(grad.square(), alpha=1 - gamma_decay)
                grad_avg.mul_(gamma_decay).add_(grad, alpha=1 - gamma_decay)

                # Compute adaptive gamma (measurement strength adapts to gradient magnitude)
                if adaptive_gamma:
                    # γ_t = γ_0 * (1 + ||∇L||²)^(-α)
                    grad_norm_sq = grad.square().mean()
                    gamma_adaptive = gamma / (1.0 + grad_norm_sq.item() + eps)
                else:
                    gamma_adaptive = gamma

                # Compute adaptive beta (exploration adapts to landscape curvature)
                if adaptive_beta:
                    # Estimate curvature from gradient variance
                    grad_var = grad_sq_avg - grad_avg.square()
                    # Higher variance -> more exploration
                    beta_adaptive = beta * (1.0 + grad_var.mean().sqrt().item())
                else:
                    beta_adaptive = beta

                # Natural gradient approximation (Fisher information)
                if natural_gradient:
                    fisher_approx = state['fisher_approx']
                    fisher_approx.mul_(0.999).add_(grad.square(), alpha=0.001)
                    precond_grad = grad / (fisher_approx.sqrt() + eps)
                else:
                    precond_grad = grad

                # Damping term: γ * (∇L)²
                # This represents measurement backaction - stronger gradients induce stronger damping
                damping_term = gamma_adaptive * grad.square()

                # Drift term: η * ∇L
                # Standard gradient descent component
                drift_term = lr * precond_grad

                # Stochastic term: β * ∇L * ε_t
                # Multiplicative noise scaled by gradient magnitude (state-dependent diffusion)
                noise = torch.randn_like(grad)
                diffusion_term = beta_adaptive * precond_grad * noise * math.sqrt(dt)

                # Combined update: θ_{t+1} = θ_t - [damping + drift]dt + diffusion
                p.add_(damping_term + drift_term, alpha=-dt)
                p.add_(diffusion_term)

        return loss


class BelavkinOptimizerSGLD(Optimizer):
    """
    Variant of Belavkin optimizer closer to Stochastic Gradient Langevin Dynamics.

    This version uses additive noise instead of multiplicative noise, making it
    more similar to SGLD while retaining the adaptive damping term.

    Update rule:
        θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*√Δt*ε

    Args:
        Similar to BelavkinOptimizer
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        dt: float = 1.0,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(lr=lr, gamma=gamma, beta=beta, dt=dt, eps=eps)
        super(BelavkinOptimizerSGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with additive noise."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            dt = group['dt']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Damping term
                damping_term = gamma * grad.square()

                # Drift term
                drift_term = lr * grad

                # Additive noise (SGLD-style)
                noise = torch.randn_like(grad)
                diffusion_term = beta * noise * math.sqrt(dt)

                # Update
                p.add_(damping_term + drift_term, alpha=-dt)
                p.add_(diffusion_term)

        return loss


class BelavkinOptimizerMinimal(Optimizer):
    """
    Minimal version of Belavkin optimizer for ablation studies.

    This removes adaptive components and additional features to isolate
    the core quantum filtering-inspired mechanism.

    Update rule:
        θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L] + β*∇L*ε

    Args:
        params: Parameters to optimize
        lr: Learning rate
        gamma: Damping factor
        beta: Exploration factor
    """

    def __init__(self, params, lr: float = 1e-3, gamma: float = 1e-4, beta: float = 1e-2):
        defaults = dict(lr=lr, gamma=gamma, beta=beta)
        super(BelavkinOptimizerMinimal, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Minimal update step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                lr = group['lr']
                gamma = group['gamma']
                beta = group['beta']

                # Three-term update
                p.add_(grad.square(), alpha=-gamma)  # Damping
                p.add_(grad, alpha=-lr)  # Drift
                p.add_(grad * torch.randn_like(grad), alpha=beta)  # Diffusion

        return loss
