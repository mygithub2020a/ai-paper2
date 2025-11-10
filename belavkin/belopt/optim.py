"""
BelOpt: Belavkin-inspired optimizer for deep learning.

Implements the discrete-time Belavkin update:
    θ_{t+1} = θ_t - (γ_t ⊙ g_t ⊙ g_t + η_t ⊙ g_t) + β_t ⊙ g_t ⊙ ϵ_t

where:
    - γ_t ≥ 0: adaptive damping (curvature/noise control)
    - η_t > 0: descent coefficient (base learning rate)
    - β_t ≥ 0: stochastic exploration scale (innovation noise)
    - ϵ_t ~ N(0, σ²I): Gaussian exploration noise
    - ⊙: element-wise multiplication
"""

import math
from typing import Optional, Callable, Dict, Any, Tuple, List

import torch
from torch.optim.optimizer import Optimizer


class BelOpt(Optimizer):
    """
    Belavkin-inspired optimizer for deep learning.

    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate (η). Default: 1e-3
        gamma0 (float): Initial damping coefficient (γ). Default: 1e-3
        beta0 (float): Initial exploration noise scale (β). Default: 0.0
        eps (float): Term added to denominator for numerical stability. Default: 1e-8
        weight_decay (float): Weight decay coefficient. Default: 0.0
        decoupled_weight_decay (bool): Whether to use decoupled weight decay (AdamW-style). Default: True
        grad_clip (Optional[float]): Gradient norm clipping value. Default: None
        update_clip (Optional[float]): Update norm clipping value. Default: None
        adaptive_gamma (bool): Whether to adapt γ using EMA of g². Default: True
        beta_1 (float): EMA coefficient for first moment (used in adaptive gamma). Default: 0.9
        beta_2 (float): EMA coefficient for second moment (used in adaptive gamma). Default: 0.999
        noise_sigma (float): Standard deviation for exploration noise. Default: 1.0
        deterministic (bool): If True, sets β=0 (no exploration noise). Default: False
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma0: float = 1e-3,
        beta0: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        grad_clip: Optional[float] = None,
        update_clip: Optional[float] = None,
        adaptive_gamma: bool = True,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        noise_sigma: float = 1.0,
        deterministic: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma0 < 0.0:
            raise ValueError(f"Invalid gamma0: {gamma0}")
        if beta0 < 0.0:
            raise ValueError(f"Invalid beta0: {beta0}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"Invalid beta_1: {beta_1}")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"Invalid beta_2: {beta_2}")

        defaults = dict(
            lr=lr,
            gamma0=gamma0,
            beta0=beta0,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            grad_clip=grad_clip,
            update_clip=update_clip,
            adaptive_gamma=adaptive_gamma,
            beta_1=beta_1,
            beta_2=beta_2,
            noise_sigma=noise_sigma,
            deterministic=deterministic,
        )
        super(BelOpt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            gamma0 = group['gamma0']
            beta0 = group['beta0']
            eps = group['eps']
            weight_decay = group['weight_decay']
            decoupled_wd = group['decoupled_weight_decay']
            grad_clip = group['grad_clip']
            update_clip = group['update_clip']
            adaptive_gamma = group['adaptive_gamma']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            noise_sigma = group['noise_sigma']
            deterministic = group['deterministic']

            # Process each parameter
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply gradient clipping if specified
                if grad_clip is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > grad_clip:
                        grad = grad * (grad_clip / (grad_norm + 1e-10))

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # EMA of gradient second moment (for adaptive gamma)
                    if adaptive_gamma:
                        state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                # Decoupled weight decay (applied before main update)
                if decoupled_wd and weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # Compute adaptive gamma if enabled
                if adaptive_gamma:
                    v = state['v']
                    # Update EMA of g²
                    v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                    # Bias correction
                    v_hat = v / (1 - beta_2 ** step)

                    # Adaptive gamma: γ_t = γ_0 / (√v_hat + eps)
                    gamma_t = gamma0 / (torch.sqrt(v_hat) + eps)
                else:
                    # Scalar gamma with optional decay
                    gamma_t = gamma0 / math.sqrt(step) if gamma0 > 0 else 0.0

                # Compute beta_t (exploration noise scale)
                # Simple inverse sqrt decay
                beta_t = beta0 / math.sqrt(step) if not deterministic else 0.0

                # Compute the main update components
                # Update = (γ_t ⊙ g_t ⊙ g_t + η_t ⊙ g_t) - β_t ⊙ g_t ⊙ ϵ_t

                # First term: γ_t ⊙ g² (damping/curvature)
                if isinstance(gamma_t, torch.Tensor):
                    damping_term = gamma_t * grad * grad
                else:
                    damping_term = gamma_t * grad * grad if gamma_t > 0 else 0.0

                # Second term: η_t ⊙ g (gradient descent)
                descent_term = lr * grad

                # Third term: β_t ⊙ g ⊙ ϵ (exploration/innovation)
                if beta_t > 0:
                    # Draw exploration noise ϵ ~ N(0, σ²I)
                    epsilon = torch.randn_like(grad) * noise_sigma
                    exploration_term = beta_t * grad * epsilon
                else:
                    exploration_term = 0.0

                # Combined update: Δθ = -(damping + descent) + exploration
                update = -(damping_term + descent_term) + exploration_term

                # Apply update clipping if specified
                if update_clip is not None:
                    if isinstance(update, torch.Tensor):
                        update_norm = torch.norm(update)
                        if update_norm > update_clip:
                            update = update * (update_clip / (update_norm + 1e-10))

                # Apply update
                p.add_(update)

        return loss

    def get_lr_scheduler_step(self) -> int:
        """Get the current step for learning rate scheduling."""
        if len(self.param_groups) > 0:
            state = self.state[self.param_groups[0]['params'][0]]
            return state.get('step', 0)
        return 0

    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for group in self.param_groups:
            group['lr'] = lr

    def set_gamma(self, gamma: float):
        """Set gamma for all parameter groups."""
        for group in self.param_groups:
            group['gamma0'] = gamma

    def set_beta(self, beta: float):
        """Set beta for all parameter groups."""
        for group in self.param_groups:
            group['beta0'] = beta
