"""
Baseline optimizer utilities for comparison benchmarks.

Provides standardized interfaces to common optimizers (SGD, Adam, RMSprop, etc.)
for fair comparison with Belavkin-inspired optimizers.
"""

from typing import Dict, Any, List
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.optimizer import Optimizer


def get_baseline_optimizer(
    params,
    optimizer_name: str,
    lr: float = 1e-3,
    **kwargs
) -> Optimizer:
    """
    Factory function to create baseline optimizers with standardized defaults.

    Args:
        params: Model parameters to optimize
        optimizer_name (str): Name of optimizer ('sgd', 'adam', 'adamw', 'rmsprop', 'sgld')
        lr (float): Learning rate
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = get_baseline_optimizer(model.parameters(), 'adam', lr=1e-3)
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return SGD(
            params,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0),
            nesterov=kwargs.get('nesterov', False),
        )

    elif optimizer_name == 'adam':
        return Adam(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0),
            amsgrad=kwargs.get('amsgrad', False),
        )

    elif optimizer_name == 'adamw':
        return AdamW(
            params,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01),
            amsgrad=kwargs.get('amsgrad', False),
        )

    elif optimizer_name == 'rmsprop':
        return RMSprop(
            params,
            lr=lr,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0),
            momentum=kwargs.get('momentum', 0),
        )

    elif optimizer_name == 'sgld':
        # Stochastic Gradient Langevin Dynamics
        return SGLD(
            params,
            lr=lr,
            temperature=kwargs.get('temperature', 1e-4),
            weight_decay=kwargs.get('weight_decay', 0),
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Choose from: sgd, adam, adamw, rmsprop, sgld"
        )


class SGLD(Optimizer):
    """
    Stochastic Gradient Langevin Dynamics optimizer.

    Implements SGLD as described in "Bayesian Learning via Stochastic Gradient
    Langevin Dynamics" (Welling & Teh, 2011).

    Update rule:
        θ_{t+1} = θ_t - η*∇L(θ) + √(2η*T) * ε

    where T is the temperature parameter and ε ~ N(0, I).

    Args:
        params: Model parameters
        lr (float): Learning rate η
        temperature (float): Temperature T for noise scaling
        weight_decay (float): Weight decay (L2 penalty)

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=1e-3, temperature=1e-4)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        temperature: float = 1e-4,
        weight_decay: float = 0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")

        defaults = dict(
            lr=lr,
            temperature=temperature,
            weight_decay=weight_decay,
        )
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            temperature = group['temperature']
            weight_decay = group['weight_decay']

            # Noise scale: √(2η*T)
            noise_scale = (2 * lr * temperature) ** 0.5

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # SGLD update: θ = θ - η*∇L + √(2ηT)*ε
                noise = torch.randn_like(p) * noise_scale
                p.add_(grad, alpha=-lr)
                p.add_(noise)

        return loss


def get_optimizer_hyperparameter_grid(optimizer_name: str) -> Dict[str, List[Any]]:
    """
    Get default hyperparameter search grid for a given optimizer.

    Args:
        optimizer_name (str): Name of optimizer

    Returns:
        Dictionary mapping hyperparameter names to lists of values to try

    Example:
        >>> grid = get_optimizer_hyperparameter_grid('adam')
        >>> print(grid['lr'])
        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    """
    base_lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    if optimizer_name.lower() == 'sgd':
        return {
            'lr': base_lrs,
            'momentum': [0.0, 0.9, 0.99],
            'weight_decay': [0, 1e-4, 1e-3],
        }

    elif optimizer_name.lower() in ['adam', 'adamw']:
        return {
            'lr': base_lrs,
            'weight_decay': [0, 1e-4, 1e-3, 1e-2],
            'betas': [(0.9, 0.999), (0.9, 0.99)],
        }

    elif optimizer_name.lower() == 'rmsprop':
        return {
            'lr': base_lrs,
            'alpha': [0.9, 0.99, 0.999],
            'weight_decay': [0, 1e-4, 1e-3],
        }

    elif optimizer_name.lower() == 'sgld':
        return {
            'lr': base_lrs,
            'temperature': [1e-5, 1e-4, 1e-3],
            'weight_decay': [0, 1e-4, 1e-3],
        }

    elif optimizer_name.lower() == 'belavkin':
        return {
            'lr': base_lrs,
            'gamma': [1e-5, 1e-4, 1e-3, 1e-2],
            'beta': [1e-3, 1e-2, 1e-1],
            'adaptive_gamma': [True, False],
        }

    else:
        return {'lr': base_lrs}
