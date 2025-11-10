"""
Benchmark Framework for Optimizer Comparison

This module provides a comprehensive benchmarking system for comparing
the Belavkin optimizer against standard baselines (SGD, Adam, RMSprop, etc.)

Features:
    - Unified training loop for all optimizers
    - Hyperparameter grid search
    - Multi-seed evaluation for statistical significance
    - Comprehensive metrics collection
    - Experiment tracking and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import time
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

import sys
sys.path.append('..')
from track1_optimizer import BelavkinOptimizer, BelavkinOptimizerSGLD, BelavkinOptimizerMinimal


class OptimizerBenchmark:
    """
    Benchmarking framework for optimizer comparison.

    Args:
        model_fn: Function that creates a fresh model instance
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on ('cuda' or 'cpu')
        criterion: Loss function
        metrics: Dictionary of metric functions
    """

    def __init__(
        self,
        model_fn: Callable,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        criterion: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.metrics = metrics or {'accuracy': self._accuracy}

        self.results = defaultdict(list)

    @staticmethod
    def _accuracy(outputs, targets):
        """Compute classification accuracy."""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def create_optimizer(
        self, optimizer_name: str, model: nn.Module, **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create optimizer instance.

        Args:
            optimizer_name: Name of optimizer
            model: Model to optimize
            **kwargs: Optimizer-specific hyperparameters

        Returns:
            Optimizer instance
        """
        lr = kwargs.get('lr', 1e-3)

        if optimizer_name == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        elif optimizer_name == 'adam':
            betas = kwargs.get('betas', (0.9, 0.999))
            return torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

        elif optimizer_name == 'adamw':
            betas = kwargs.get('betas', (0.9, 0.999))
            weight_decay = kwargs.get('weight_decay', 0.01)
            return torch.optim.AdamW(
                model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )

        elif optimizer_name == 'rmsprop':
            alpha = kwargs.get('alpha', 0.99)
            return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)

        elif optimizer_name == 'belavkin':
            gamma = kwargs.get('gamma', 1e-4)
            beta = kwargs.get('beta', 1e-2)
            adaptive_gamma = kwargs.get('adaptive_gamma', False)
            adaptive_beta = kwargs.get('adaptive_beta', False)
            return BelavkinOptimizer(
                model.parameters(),
                lr=lr,
                gamma=gamma,
                beta=beta,
                adaptive_gamma=adaptive_gamma,
                adaptive_beta=adaptive_beta,
            )

        elif optimizer_name == 'belavkin_sgld':
            gamma = kwargs.get('gamma', 1e-4)
            beta = kwargs.get('beta', 1e-2)
            return BelavkinOptimizerSGLD(
                model.parameters(), lr=lr, gamma=gamma, beta=beta
            )

        elif optimizer_name == 'belavkin_minimal':
            gamma = kwargs.get('gamma', 1e-4)
            beta = kwargs.get('beta', 1e-2)
            return BelavkinOptimizerMinimal(
                model.parameters(), lr=lr, gamma=gamma, beta=beta
            )

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train_epoch(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            model: Model to train
            optimizer: Optimizer

        Returns:
            Dictionary of training metrics
        """
        model.train()
        total_loss = 0.0
        total_samples = 0
        metric_values = defaultdict(float)

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute metrics
            for metric_name, metric_fn in self.metrics.items():
                metric_values[metric_name] += metric_fn(outputs, targets) * batch_size

        # Average over epoch
        results = {'loss': total_loss / total_samples}
        for metric_name, value in metric_values.items():
            results[metric_name] = value / total_samples

        return results

    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            model: Model to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_values = defaultdict(float)

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Compute metrics
                for metric_name, metric_fn in self.metrics.items():
                    metric_values[metric_name] += metric_fn(outputs, targets) * batch_size

        # Average
        results = {'loss': total_loss / total_samples}
        for metric_name, value in metric_values.items():
            results[metric_name] = value / total_samples

        return results

    def run_experiment(
        self,
        optimizer_name: str,
        n_epochs: int = 100,
        seed: int = 42,
        log_interval: int = 10,
        early_stopping_patience: Optional[int] = None,
        target_accuracy: Optional[float] = None,
        **optimizer_kwargs,
    ) -> Dict:
        """
        Run a single experiment with specified optimizer.

        Args:
            optimizer_name: Name of optimizer
            n_epochs: Number of epochs
            seed: Random seed
            log_interval: Logging interval
            early_stopping_patience: Early stopping patience
            target_accuracy: Stop when this accuracy is reached
            **optimizer_kwargs: Optimizer hyperparameters

        Returns:
            Dictionary of results
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create fresh model and optimizer
        model = self.model_fn().to(self.device)
        optimizer = self.create_optimizer(optimizer_name, model, **optimizer_kwargs)

        # Tracking
        history = defaultdict(list)
        best_test_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(model, optimizer)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))

            # Evaluate
            test_metrics = self.evaluate(model)
            history['test_loss'].append(test_metrics['loss'])
            history['test_accuracy'].append(test_metrics.get('accuracy', 0.0))

            # Track best
            if test_metrics.get('accuracy', 0.0) > best_test_acc:
                best_test_acc = test_metrics.get('accuracy', 0.0)
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Logging
            if (epoch + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Test Loss: {test_metrics['loss']:.4f} | "
                    f"Test Acc: {test_metrics.get('accuracy', 0.0):.4f}"
                )

            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Target accuracy reached
            if target_accuracy and test_metrics.get('accuracy', 0.0) >= target_accuracy:
                print(f"Target accuracy {target_accuracy} reached at epoch {epoch+1}")
                break

        # Compute wall-clock time
        elapsed_time = time.time() - start_time

        # Compute convergence metrics
        epochs_to_90 = self._epochs_to_target(history['test_accuracy'], 0.90)
        epochs_to_95 = self._epochs_to_target(history['test_accuracy'], 0.95)
        epochs_to_99 = self._epochs_to_target(history['test_accuracy'], 0.99)

        results = {
            'optimizer': optimizer_name,
            'seed': seed,
            'optimizer_kwargs': optimizer_kwargs,
            'history': dict(history),
            'best_test_accuracy': best_test_acc,
            'best_epoch': best_epoch,
            'final_test_accuracy': history['test_accuracy'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'train_test_gap': history['train_accuracy'][-1] - history['test_accuracy'][-1],
            'total_epochs': len(history['train_loss']),
            'elapsed_time': elapsed_time,
            'time_per_epoch': elapsed_time / len(history['train_loss']),
            'epochs_to_90': epochs_to_90,
            'epochs_to_95': epochs_to_95,
            'epochs_to_99': epochs_to_99,
        }

        return results

    @staticmethod
    def _epochs_to_target(accuracies: List[float], target: float) -> Optional[int]:
        """Find first epoch where accuracy reaches target."""
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i + 1
        return None

    def grid_search(
        self,
        optimizer_configs: Dict[str, Dict],
        n_seeds: int = 3,
        n_epochs: int = 100,
        log_interval: int = 10,
    ) -> List[Dict]:
        """
        Perform grid search over optimizer configurations.

        Args:
            optimizer_configs: Dict mapping optimizer names to config dicts
            n_seeds: Number of random seeds
            n_epochs: Number of epochs per run
            log_interval: Logging interval

        Returns:
            List of all experiment results

        Example:
            configs = {
                'adam': {'lr': [1e-4, 1e-3, 1e-2]},
                'belavkin': {
                    'lr': [1e-4, 1e-3],
                    'gamma': [1e-5, 1e-4],
                    'beta': [1e-2, 1e-1]
                }
            }
            results = benchmark.grid_search(configs, n_seeds=3)
        """
        all_results = []

        for optimizer_name, config in optimizer_configs.items():
            print(f"\n{'='*60}")
            print(f"Testing optimizer: {optimizer_name}")
            print(f"{'='*60}")

            # Generate all combinations of hyperparameters
            param_grid = self._expand_grid(config)

            for param_set in param_grid:
                print(f"\nHyperparameters: {param_set}")

                # Run with multiple seeds
                for seed in range(n_seeds):
                    print(f"  Seed {seed + 1}/{n_seeds}")

                    result = self.run_experiment(
                        optimizer_name=optimizer_name,
                        n_epochs=n_epochs,
                        seed=seed,
                        log_interval=log_interval,
                        **param_set,
                    )

                    all_results.append(result)

        return all_results

    @staticmethod
    def _expand_grid(config: Dict) -> List[Dict]:
        """Expand hyperparameter grid."""
        keys = list(config.keys())
        values = list(config.values())

        # Handle non-list values
        values = [[v] if not isinstance(v, list) else v for v in values]

        # Generate all combinations
        import itertools

        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    def summarize_results(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics across experiments.

        Args:
            results: List of experiment results

        Returns:
            Dictionary of summary statistics
        """
        # Group by optimizer
        by_optimizer = defaultdict(list)
        for r in results:
            key = (r['optimizer'], tuple(sorted(r['optimizer_kwargs'].items())))
            by_optimizer[key].append(r)

        summary = {}
        for (opt_name, params), runs in by_optimizer.items():
            # Compute mean and std across seeds
            best_accs = [r['best_test_accuracy'] for r in runs]
            final_accs = [r['final_test_accuracy'] for r in runs]
            times = [r['elapsed_time'] for r in runs]

            summary[(opt_name, dict(params))] = {
                'n_runs': len(runs),
                'best_accuracy_mean': np.mean(best_accs),
                'best_accuracy_std': np.std(best_accs),
                'final_accuracy_mean': np.mean(final_accs),
                'final_accuracy_std': np.std(final_accs),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
            }

        return summary
