"""
Benchmark suite for comparing optimizers.

Compares Belavkin optimizer against:
- SGD (with momentum)
- Adam
- RMSprop
- AdamW
- SGLD (Stochastic Gradient Langevin Dynamics)

Metrics tracked:
- Convergence speed (steps to target accuracy)
- Final performance (best validation accuracy)
- Stability (variance across seeds)
- Sample efficiency
- Computational cost
- Generalization (train-test gap)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path

from belavkin_ml.optimizer import BelavkinOptimizer, AdaptiveBelavkinOptimizer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    # Optimizers to compare
    optimizers: List[str] = field(
        default_factory=lambda: ["sgd", "adam", "rmsprop", "adamw", "belavkin", "adaptive_belavkin"]
    )

    # Hyperparameter grid for each optimizer
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

    # Belavkin-specific hyperparameters
    gammas: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2])
    betas: List[float] = field(default_factory=lambda: [1e-3, 1e-2, 1e-1])

    # Training settings
    n_epochs: int = 100
    batch_size: int = 32
    n_seeds: int = 3
    early_stopping_patience: int = 20

    # Metrics
    target_accuracies: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])

    # Computational
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    # Output
    save_dir: Path = Path("experiments/track1_optimizer/benchmarks")
    save_checkpoints: bool = False
    log_interval: int = 10


class OptimizerBenchmark:
    """
    Benchmark suite for optimizer comparison.

    Example:
        >>> config = BenchmarkConfig()
        >>> benchmark = OptimizerBenchmark(config)
        >>> results = benchmark.run(model_fn, train_loader, test_loader)
        >>> benchmark.save_results(results)
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

    def create_optimizer(
        self,
        optimizer_name: str,
        model: nn.Module,
        lr: float,
        gamma: Optional[float] = None,
        beta: Optional[float] = None,
        **kwargs
    ):
        """Create an optimizer instance."""

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                **kwargs
            )

        elif optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                **kwargs
            )

        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                alpha=0.99,
                **kwargs
            )

        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                **kwargs
            )

        elif optimizer_name == "sgld":
            # Stochastic Gradient Langevin Dynamics
            # Similar to SGD with added Gaussian noise
            # Note: This is a simplified implementation
            return SGLDOptimizer(
                model.parameters(),
                lr=lr,
                temperature=beta if beta is not None else 1e-2,
                **kwargs
            )

        elif optimizer_name == "belavkin":
            if gamma is None or beta is None:
                raise ValueError("Belavkin optimizer requires gamma and beta")
            return BelavkinOptimizer(
                model.parameters(),
                lr=lr,
                gamma=gamma,
                beta=beta,
                **kwargs
            )

        elif optimizer_name == "adaptive_belavkin":
            if gamma is None or beta is None:
                raise ValueError("Adaptive Belavkin optimizer requires gamma and beta")
            return AdaptiveBelavkinOptimizer(
                model.parameters(),
                lr=lr,
                gamma=gamma,
                beta=beta,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train_epoch(
        self,
        model: nn.Module,
        optimizer,
        criterion,
        train_loader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(
        self,
        model: nn.Module,
        criterion,
        test_loader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        """Evaluate on test set."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def run_single_config(
        self,
        model_fn,
        optimizer_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float,
        gamma: Optional[float] = None,
        beta: Optional[float] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run a single optimizer configuration."""

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        model = model_fn().to(self.config.device)
        criterion = nn.CrossEntropyLoss()

        # Create optimizer
        optimizer = self.create_optimizer(
            optimizer_name, model, lr, gamma, beta
        )

        # Training metrics
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        steps_to_target = {target: None for target in self.config.target_accuracies}
        wall_times = []

        best_test_acc = 0.0
        patience_counter = 0
        total_steps = 0

        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(
                model, optimizer, criterion, train_loader, self.config.device
            )

            # Evaluate
            test_loss, test_acc = self.evaluate(
                model, criterion, test_loader, self.config.device
            )

            epoch_time = time.time() - epoch_start

            # Record metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            wall_times.append(epoch_time)

            total_steps += len(train_loader)

            # Check target accuracies
            for target in self.config.target_accuracies:
                if steps_to_target[target] is None and test_acc >= target:
                    steps_to_target[target] = total_steps

            # Early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        total_time = time.time() - start_time

        # Compute final metrics
        results = {
            'optimizer': optimizer_name,
            'lr': lr,
            'gamma': gamma,
            'beta': beta,
            'seed': seed,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_accs[-1],
            'steps_to_target': steps_to_target,
            'total_time': total_time,
            'avg_epoch_time': np.mean(wall_times),
            'train_test_gap': train_accs[-1] - test_accs[-1],
            'n_epochs_trained': len(train_losses),
        }

        return results

    def run(
        self,
        model_fn,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, List[Dict]]:
        """
        Run full benchmark suite.

        Args:
            model_fn: Function that returns a fresh model instance
            train_loader: Training data loader
            test_loader: Test data loader

        Returns:
            results: Dictionary mapping optimizer names to list of results
        """

        all_results = {opt: [] for opt in self.config.optimizers}

        total_configs = 0
        for opt in self.config.optimizers:
            n_lrs = len(self.config.learning_rates)
            if opt in ["belavkin", "adaptive_belavkin", "sgld"]:
                n_lrs *= len(self.config.gammas) * len(self.config.betas)
            total_configs += n_lrs * self.config.n_seeds

        print(f"Running {total_configs} total configurations...")

        config_idx = 0

        for optimizer_name in self.config.optimizers:
            print(f"\n{'='*60}")
            print(f"Optimizer: {optimizer_name}")
            print(f"{'='*60}")

            for lr in self.config.learning_rates:
                if optimizer_name in ["belavkin", "adaptive_belavkin", "sgld"]:
                    # Grid search over gamma and beta
                    for gamma in self.config.gammas:
                        for beta in self.config.betas:
                            for seed in range(self.config.n_seeds):
                                config_idx += 1
                                print(f"[{config_idx}/{total_configs}] "
                                      f"lr={lr:.0e}, gamma={gamma:.0e}, beta={beta:.0e}, seed={seed}")

                                result = self.run_single_config(
                                    model_fn, optimizer_name, train_loader, test_loader,
                                    lr, gamma, beta, seed
                                )
                                all_results[optimizer_name].append(result)

                else:
                    # Standard optimizers: only search over lr
                    for seed in range(self.config.n_seeds):
                        config_idx += 1
                        print(f"[{config_idx}/{total_configs}] lr={lr:.0e}, seed={seed}")

                        result = self.run_single_config(
                            model_fn, optimizer_name, train_loader, test_loader,
                            lr, None, None, seed
                        )
                        all_results[optimizer_name].append(result)

        return all_results

    def save_results(self, results: Dict[str, List[Dict]], filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        save_path = self.config.save_dir / filename

        # Convert to serializable format
        serializable_results = {}
        for opt_name, opt_results in results.items():
            serializable_results[opt_name] = []
            for result in opt_results:
                # Convert numpy types to Python types
                clean_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.integer, np.floating)):
                        clean_result[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        clean_result[key] = value.tolist()
                    elif isinstance(value, list):
                        clean_result[key] = [
                            float(x) if isinstance(x, (np.integer, np.floating)) else x
                            for x in value
                        ]
                    else:
                        clean_result[key] = value
                serializable_results[opt_name].append(clean_result)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {save_path}")


class SGLDOptimizer(torch.optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    Implements: θ_{t+1} = θ_t - η*∇L(θ) + √(2η*T)*ε
    where T is temperature and ε ~ N(0, I)

    This serves as a theoretical baseline for comparison with Belavkin.
    """

    def __init__(self, params, lr=1e-3, temperature=1e-2, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")

        defaults = dict(lr=lr, temperature=temperature, weight_decay=weight_decay)
        super(SGLDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            temperature = group['temperature']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Deterministic gradient step
                p.add_(grad, alpha=-lr)

                # Stochastic Langevin noise
                noise = torch.randn_like(p)
                noise_scale = torch.sqrt(torch.tensor(2 * lr * temperature))
                p.add_(noise, alpha=noise_scale.item())

        return loss
