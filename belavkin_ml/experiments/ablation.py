"""
Ablation study framework for Belavkin optimizer.

Systematically tests the contribution of each component:
- Gradient-dependent damping
- Multiplicative vs additive noise
- Adaptive vs fixed hyperparameters
- Individual adaptive mechanisms
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Any, Callable
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json

from belavkin_ml.optimizer import BelavkinOptimizer, AdaptiveBelavkinOptimizer


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    components_to_ablate: List[str]
    base_lr: float = 1e-3
    base_gamma: float = 1e-4
    base_beta: float = 1e-2
    n_epochs: int = 100
    n_seeds: int = 3
    save_dir: Path = Path("experiments/track1_optimizer/ablations")


class AblationOptimizers:
    """Collection of ablated optimizer variants."""

    @staticmethod
    def get_optimizer(
        variant: str,
        params,
        lr: float,
        gamma: float,
        beta: float,
    ) -> Optimizer:
        """
        Get ablated optimizer variant.

        Variants:
        - 'full': Full Belavkin optimizer
        - 'no_damping': Set gamma=0
        - 'no_exploration': Set beta=0
        - 'additive_noise': Replace multiplicative with additive noise
        - 'no_adaptation': Disable all adaptation
        - 'no_adaptive_gamma': Disable gamma adaptation only
        - 'no_adaptive_beta': Disable beta adaptation only
        """

        if variant == 'full':
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=beta,
                adaptive_gamma=True, adaptive_beta=True
            )

        elif variant == 'no_damping':
            return BelavkinOptimizer(
                params, lr=lr, gamma=0.0, beta=beta,
                adaptive_gamma=False, adaptive_beta=True
            )

        elif variant == 'no_exploration':
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=0.0,
                adaptive_gamma=True, adaptive_beta=False
            )

        elif variant == 'additive_noise':
            return AdditiveNoiseOptimizer(
                params, lr=lr, gamma=gamma, beta=beta
            )

        elif variant == 'no_adaptation':
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=beta,
                adaptive_gamma=False, adaptive_beta=False
            )

        elif variant == 'no_adaptive_gamma':
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=beta,
                adaptive_gamma=False, adaptive_beta=True
            )

        elif variant == 'no_adaptive_beta':
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=beta,
                adaptive_gamma=True, adaptive_beta=False
            )

        elif variant == 'only_damping':
            # Only damping, no exploration
            return BelavkinOptimizer(
                params, lr=lr, gamma=gamma, beta=0.0,
                adaptive_gamma=False, adaptive_beta=False
            )

        elif variant == 'only_exploration':
            # Only exploration, no damping
            return BelavkinOptimizer(
                params, lr=lr, gamma=0.0, beta=beta,
                adaptive_gamma=False, adaptive_beta=False
            )

        elif variant == 'sgd_baseline':
            # Standard SGD for comparison
            return torch.optim.SGD(params, lr=lr, momentum=0.9)

        else:
            raise ValueError(f"Unknown variant: {variant}")


class AdditiveNoiseOptimizer(Optimizer):
    """
    Belavkin optimizer with additive instead of multiplicative noise.

    Update: θ_{t+1} = θ_t - [γ*(∇L)² + η*∇L]Δt + β*ε

    (Compare to multiplicative: β*∇L*ε)
    """

    def __init__(self, params, lr=1e-3, gamma=1e-4, beta=1e-2):
        defaults = dict(lr=lr, gamma=gamma, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Damping term
                grad_squared = grad * grad

                # Deterministic update
                deterministic_update = -(gamma * grad_squared + lr * grad)

                # ADDITIVE noise (not multiplicative)
                noise = torch.randn_like(grad)
                stochastic_update = beta * noise

                # Combined update
                update = deterministic_update + stochastic_update
                p.add_(update)

        return loss


class AblationStudy:
    """
    Framework for running ablation studies.

    Example:
        >>> config = AblationConfig(
        ...     components_to_ablate=['no_damping', 'no_exploration', 'additive_noise']
        ... )
        >>> study = AblationStudy(config)
        >>> results = study.run(model_fn, train_loader, test_loader)
        >>> study.analyze_results(results)
    """

    def __init__(self, config: AblationConfig):
        self.config = config
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

    def run_single_variant(
        self,
        variant: str,
        model_fn: Callable,
        train_loader,
        test_loader,
        device: str = "cpu",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run a single ablated variant."""

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model and optimizer
        model = model_fn().to(device)
        optimizer = AblationOptimizers.get_optimizer(
            variant, model.parameters(),
            self.config.base_lr,
            self.config.base_gamma,
            self.config.base_beta,
        )

        criterion = nn.CrossEntropyLoss()

        # Training loop
        train_accs = []
        test_accs = []

        for epoch in range(self.config.n_epochs):
            # Train
            model.train()
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = correct / total

            # Test
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = correct / total

            train_accs.append(train_acc)
            test_accs.append(test_acc)

        return {
            'variant': variant,
            'seed': seed,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'final_test_acc': test_accs[-1],
            'best_test_acc': max(test_accs),
        }

    def run(
        self,
        model_fn: Callable,
        train_loader,
        test_loader,
        device: str = "cpu",
    ) -> Dict[str, List[Dict]]:
        """Run full ablation study."""

        # Always include full model and SGD baseline
        variants = ['full', 'sgd_baseline'] + self.config.components_to_ablate

        results = {variant: [] for variant in variants}

        total = len(variants) * self.config.n_seeds
        count = 0

        for variant in variants:
            print(f"\nTesting variant: {variant}")

            for seed in range(self.config.n_seeds):
                count += 1
                print(f"  [{count}/{total}] Seed {seed}")

                result = self.run_single_variant(
                    variant, model_fn, train_loader, test_loader, device, seed
                )
                results[variant].append(result)

        return results

    def analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Analyze ablation study results."""

        analysis = {}

        for variant, variant_results in results.items():
            test_accs = [r['best_test_acc'] for r in variant_results]

            analysis[variant] = {
                'mean_acc': np.mean(test_accs),
                'std_acc': np.std(test_accs),
                'min_acc': np.min(test_accs),
                'max_acc': np.max(test_accs),
            }

        # Compute relative performance vs full model
        full_mean = analysis['full']['mean_acc']

        for variant in analysis:
            if variant != 'full':
                analysis[variant]['relative_to_full'] = \
                    analysis[variant]['mean_acc'] - full_mean

        return analysis

    def save_results(self, results: Dict, analysis: Dict, filename: str = "ablation_results.json"):
        """Save results and analysis."""

        # Convert to JSON-serializable format
        json_results = {}
        for variant, variant_results in results.items():
            json_results[variant] = []
            for result in variant_results:
                json_result = {k: v if not isinstance(v, list) else
                               [float(x) for x in v]
                               for k, v in result.items()}
                json_results[variant].append(json_result)

        output = {
            'results': json_results,
            'analysis': analysis,
            'config': {
                'components_to_ablate': self.config.components_to_ablate,
                'base_lr': self.config.base_lr,
                'base_gamma': self.config.base_gamma,
                'base_beta': self.config.base_beta,
                'n_epochs': self.config.n_epochs,
                'n_seeds': self.config.n_seeds,
            }
        }

        save_path = self.config.save_dir / filename

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {save_path}")

    def print_summary(self, analysis: Dict):
        """Print summary of ablation study."""

        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)

        # Sort by performance
        sorted_variants = sorted(
            analysis.items(),
            key=lambda x: x[1]['mean_acc'],
            reverse=True
        )

        print(f"\n{'Variant':<25} {'Mean Acc':<12} {'Std':<10} {'vs Full':<10}")
        print("-"*70)

        for variant, stats in sorted_variants:
            rel = stats.get('relative_to_full', 0.0)
            rel_str = f"{rel:+.4f}" if variant != 'full' else "---"

            print(f"{variant:<25} "
                  f"{stats['mean_acc']:.4f}      "
                  f"{stats['std_acc']:.4f}    "
                  f"{rel_str}")

        print("="*70)
