"""
Belavkin-inspired optimizers for neural network training.

The core optimizer implements the update rule:
    dθ = -[γ * (∇L(θ))² + η * ∇L(θ)] dt + β * ∇L(θ) * dε_t

where:
- θ: Network parameters (analogue of quantum state)
- ∇L(θ): Loss gradient (analogue of measurement signal)
- γ: Adaptive damping factor (analogue of measurement strength)
- η: Learning rate (drift coefficient)
- β: Stochastic exploration factor (diffusion coefficient)
- ε_t: Gaussian noise term (measurement uncertainty)
"""

from belavkin_ml.optimizer.belavkin import BelavkinOptimizer
from belavkin_ml.optimizer.adaptive import AdaptiveBelavkinOptimizer

__all__ = ["BelavkinOptimizer", "AdaptiveBelavkinOptimizer"]
