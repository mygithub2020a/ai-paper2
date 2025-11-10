"""
Adaptive Belavkin Optimizer

Implements advanced adaptive mechanisms for the Belavkin optimizer,
including:
- Automatic learning rate scheduling
- Gradient-aware damping adaptation
- Loss landscape curvature estimation
- Parameter-wise adaptation
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import math


class AdaptiveBelavkinOptimizer(Optimizer):
    """
    Adaptive Belavkin optimizer with automatic hyperparameter tuning.

    This variant automatically adjusts γ, β, and learning rate based on:
    1. Gradient statistics (norm, variance)
    2. Loss landscape curvature (approximate Hessian diagonal)
    3. Training progress (loss reduction rate)

    Args:
        params: Model parameters
        lr: Initial learning rate (default: 1e-3)
        gamma: Initial damping factor (default: 1e-4)
        beta: Initial exploration factor (default: 1e-2)
        dt: Time step (default: 1.0)
        adapt_lr: Enable learning rate adaptation (default: True)
        adapt_gamma: Enable damping adaptation (default: True)
        adapt_beta: Enable exploration adaptation (default: True)
        lr_decay: Learning rate decay rate (default: 0.999)
        gamma_min: Minimum gamma value (default: 1e-6)
        gamma_max: Maximum gamma value (default: 1e-2)
        beta_min: Minimum beta value (default: 1e-4)
        beta_max: Maximum beta value (default: 1e-1)
        curvature_window: Window size for curvature estimation (default: 10)
        weight_decay: Weight decay (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 1e-4,
        beta: float = 1e-2,
        dt: float = 1.0,
        adapt_lr: bool = True,
        adapt_gamma: bool = True,
        adapt_beta: bool = True,
        lr_decay: float = 0.999,
        gamma_min: float = 1e-6,
        gamma_max: float = 1e-2,
        beta_min: float = 1e-4,
        beta_max: float = 1e-1,
        curvature_window: int = 10,
        weight_decay: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma: {gamma}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            beta=beta,
            dt=dt,
            adapt_lr=adapt_lr,
            adapt_gamma=adapt_gamma,
            adapt_beta=adapt_beta,
            lr_decay=lr_decay,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            curvature_window=curvature_window,
            weight_decay=weight_decay,
        )
        super(AdaptiveBelavkinOptimizer, self).__init__(params, defaults)

        # Global state for loss tracking
        self.loss_history = []
        self.grad_norm_history = []

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with adaptive hyperparameters.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            loss: The loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if loss is not None:
                self.loss_history.append(loss.item())

        # Compute global gradient norm for adaptation
        total_grad_norm = 0.0
        total_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_grad_norm += torch.sum(p.grad ** 2).item()
                    total_params += p.numel()

        if total_params > 0:
            total_grad_norm = math.sqrt(total_grad_norm)
            self.grad_norm_history.append(total_grad_norm)

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            dt = group['dt']
            adapt_lr = group['adapt_lr']
            adapt_gamma = group['adapt_gamma']
            adapt_beta = group['adapt_beta']
            lr_decay = group['lr_decay']
            gamma_min = group['gamma_min']
            gamma_max = group['gamma_max']
            beta_min = group['beta_min']
            beta_max = group['beta_max']
            curvature_window = group['curvature_window']
            weight_decay = group['weight_decay']

            # Adaptive learning rate based on gradient norm stability
            if adapt_lr and len(self.grad_norm_history) > 5:
                # If gradient norms are stable, can increase lr slightly
                # If gradient norms are volatile, decrease lr
                recent_grads = self.grad_norm_history[-5:]
                grad_std = torch.std(torch.tensor(recent_grads)).item()
                grad_mean = torch.mean(torch.tensor(recent_grads)).item()

                if grad_mean > 0:
                    grad_cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation
                    # High CV -> reduce lr, low CV -> can increase lr
                    if grad_cv > 0.5:
                        lr = lr * 0.95
                    elif grad_cv < 0.1:
                        lr = lr * 1.01
                else:
                    lr = lr * lr_decay

                group['lr'] = lr

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_avg'] = torch.zeros_like(p)
                    state['grad_sq_avg'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    state['gamma_t'] = gamma
                    state['beta_t'] = beta

                state['step'] += 1
                grad_avg = state['grad_avg']
                grad_sq_avg = state['grad_sq_avg']
                prev_grad = state['prev_grad']

                # Update running statistics
                decay = 0.99
                grad_avg.mul_(decay).add_(grad, alpha=1 - decay)
                grad_sq_avg.mul_(decay).addcmul_(grad, grad, value=1 - decay)

                # Adaptive gamma based on gradient magnitude
                if adapt_gamma:
                    # Strong gradients -> higher damping (measurement backaction)
                    # Weak gradients -> lower damping
                    grad_norm_sq = torch.sum(grad ** 2)

                    # Use exponential scaling with clipping
                    alpha = 0.5  # Sensitivity parameter
                    gamma_t = gamma * (1 + grad_norm_sq) ** alpha
                    gamma_t = torch.clamp(
                        torch.tensor(gamma_t),
                        min=gamma_min,
                        max=gamma_max
                    ).item()

                    state['gamma_t'] = gamma_t
                else:
                    state['gamma_t'] = gamma

                # Adaptive beta based on loss landscape curvature
                if adapt_beta and state['step'] > 1:
                    # Estimate curvature using gradient differences
                    # High curvature -> lower exploration
                    # Low curvature -> higher exploration
                    grad_diff = grad - prev_grad
                    curvature_proxy = torch.sum(grad_diff ** 2) / (torch.sum(grad ** 2) + 1e-8)

                    # Inverse relationship: high curvature -> low beta
                    beta_t = beta / (1 + curvature_proxy.sqrt())
                    beta_t = torch.clamp(
                        beta_t,
                        min=beta_min,
                        max=beta_max
                    ).item()

                    state['beta_t'] = beta_t
                else:
                    state['beta_t'] = beta

                # Store current gradient for next step
                prev_grad.copy_(grad)

                gamma_t = state['gamma_t']
                beta_t = state['beta_t']

                # Belavkin update
                grad_squared = grad * grad

                # Deterministic component
                deterministic_update = -(gamma_t * grad_squared + lr * grad) * dt

                # Stochastic component (gradient-dependent noise)
                noise = torch.randn_like(grad)
                stochastic_update = beta_t * grad * math.sqrt(dt) * noise

                # Apply update
                update = deterministic_update + stochastic_update
                p.add_(update)

        return loss

    def get_adaptation_stats(self):
        """
        Returns statistics about adaptive hyperparameters.

        Returns:
            dict: Dictionary containing adaptation statistics
        """
        stats = {
            'loss_history': self.loss_history[-100:],  # Last 100 losses
            'grad_norm_history': self.grad_norm_history[-100:],
            'current_params': {}
        }

        for group_idx, group in enumerate(self.param_groups):
            stats['current_params'][f'group_{group_idx}'] = {
                'lr': group['lr'],
                'gamma': group['gamma'],
                'beta': group['beta'],
            }

            # Get parameter-wise stats
            param_stats = []
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    param_stats.append({
                        'gamma_t': state.get('gamma_t', group['gamma']),
                        'beta_t': state.get('beta_t', group['beta']),
                        'step': state['step'],
                    })

            if param_stats:
                stats['current_params'][f'group_{group_idx}']['param_wise'] = param_stats

        return stats
