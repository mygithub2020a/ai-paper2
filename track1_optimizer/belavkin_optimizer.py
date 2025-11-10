"""
Belavkin-Inspired Optimizer for PyTorch

Implementation of a novel neural network optimizer derived from quantum filtering
principles in the Belavkin equation framework.

The update rule is inspired by the Belavkin equation:
    dψ_t = -[(1/2)L*L + (i/ℏ)H]ψ_t dt + Lψ_t dy_t

Translated to parameter optimization:
    dθ = -[γ * (∇L(θ))² + η * ∇L(θ)] dt + β * ∇L(θ) * dε_t

Where:
    - θ: Network parameters (analogue of quantum state)
    - ∇L(θ): Loss gradient (analogue of measurement signal)
    - γ: Adaptive damping factor (analogue of measurement strength)
    - η: Learning rate (drift coefficient)
    - β: Stochastic exploration factor (diffusion coefficient)
    - ε_t: Gaussian noise term (measurement uncertainty)
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Callable
import math


class BelavkinOptimizer(Optimizer):
    """
    Belavkin-inspired optimizer with quantum filtering characteristics.

    This optimizer implements a novel update rule derived from the Belavkin
    quantum filtering equation. It features:
    - Gradient-dependent damping (measurement backaction)
    - Multiplicative noise (state-dependent diffusion)
    - Optional adaptive parameter tuning

    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Learning rate η (default: 1e-3)
        gamma (float): Damping factor γ (default: 1e-4)
        beta (float): Exploration factor β (default: 1e-2)
        adaptive_gamma (bool): Adapt γ based on gradient statistics (default: False)
        adaptive_beta (bool): Adapt β based on loss landscape curvature (default: False)
        gamma_decay (float): Decay rate for adaptive gamma (default: 0.5)
        grad_clip (float): Gradient clipping threshold (default: 10.0)
        eps (float): Numerical stability constant (default: 1e-8)
        weight_decay (float): L2 regularization coefficient (default: 0.0)

    Example:
        >>> optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, gamma=1e-4)
        >>> optimizer.zero_grad()
        >>> loss = criterion(model(input), target)
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        adaptive_gamma: bool = False,
        adaptive_beta: bool = False,
        gamma_decay: float = 0.5,
        grad_clip: float = 10.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= gamma_decay <= 1.0:
            raise ValueError(f"Invalid gamma_decay value: {gamma_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            adaptive_gamma=adaptive_gamma,
            adaptive_beta=adaptive_beta,
            gamma_decay=gamma_decay,
            grad_clip=grad_clip,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(BelavkinOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BelavkinOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Implements the update rule:
            θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*∇L*√Δt*ε

        Where:
            - First term: Gradient-dependent damping (measurement backaction)
            - Second term: Standard gradient descent
            - Third term: Multiplicative noise (quantum diffusion)

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (float, optional): The loss value if closure is provided.
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
            grad_clip = group['grad_clip']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Gradient clipping for numerical stability
                if grad_clip > 0:
                    grad_norm = torch.norm(grad)
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / (grad_norm + eps))

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['grad_sq_ema'] = torch.zeros_like(p)  # Running average of grad²
                    param_state['grad_norm_history'] = []  # For adaptive gamma

                param_state['step'] += 1
                step = param_state['step']

                # Compute gradient squared (for damping term)
                grad_sq = grad * grad

                # Update running statistics for adaptive parameters
                param_state['grad_sq_ema'].mul_(0.9).add_(grad_sq, alpha=0.1)

                # Adaptive gamma: γ_t = γ_0 * (1 + ||∇L||²)^(-α)
                if adaptive_gamma:
                    grad_norm = torch.norm(grad).item()
                    param_state['grad_norm_history'].append(grad_norm)

                    # Keep only recent history
                    if len(param_state['grad_norm_history']) > 100:
                        param_state['grad_norm_history'].pop(0)

                    # Adapt gamma based on gradient norm
                    gamma_adapted = gamma * (1.0 + grad_norm**2) ** (-gamma_decay)
                    gamma = max(gamma_adapted, eps)  # Ensure positivity

                # Adaptive beta: reduce exploration as training progresses
                if adaptive_beta:
                    # Simple decay schedule
                    beta_adapted = beta * (1.0 / math.sqrt(1.0 + step / 1000.0))
                    beta = max(beta_adapted, eps)

                # === BELAVKIN UPDATE RULE ===
                # dθ = -[γ * (∇L)² + η * ∇L] dt + β * ∇L * dε_t

                # Term 1: Damping term γ*(∇L)²
                # This represents measurement backaction - stronger gradients induce stronger damping
                damping_term = gamma * grad_sq

                # Term 2: Drift term η*∇L (standard gradient descent)
                drift_term = lr * grad

                # Term 3: Stochastic term β*∇L*ε (multiplicative noise)
                # Noise scaled by gradient magnitude (state-dependent diffusion)
                noise = torch.randn_like(grad)
                stochastic_term = beta * grad * noise * math.sqrt(lr)  # √Δt scaling

                # Combined update
                # Note: We use negative gradient for minimization
                update = damping_term + drift_term - stochastic_term

                # Apply update
                p.add_(-update)

        return loss

    def get_adaptive_params(self):
        """
        Returns current adaptive parameter values for monitoring.

        Returns:
            dict: Dictionary containing gamma and beta values for each parameter group.
        """
        adaptive_info = {}
        for i, group in enumerate(self.param_groups):
            adaptive_info[f'group_{i}'] = {
                'gamma': group['gamma'],
                'beta': group['beta'],
            }
        return adaptive_info

    def get_state_statistics(self):
        """
        Returns statistics about the optimizer state for analysis.

        Returns:
            dict: Dictionary containing mean gradient norms, steps, etc.
        """
        stats = {
            'mean_grad_norm': 0.0,
            'max_grad_norm': 0.0,
            'total_steps': 0,
        }

        total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = torch.norm(p.grad).item()
                    stats['mean_grad_norm'] += grad_norm
                    stats['max_grad_norm'] = max(stats['max_grad_norm'], grad_norm)
                    total_params += 1

                param_state = self.state.get(p, {})
                if 'step' in param_state:
                    stats['total_steps'] = max(stats['total_steps'], param_state['step'])

        if total_params > 0:
            stats['mean_grad_norm'] /= total_params

        return stats
