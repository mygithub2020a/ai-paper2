"""
Optimizer benchmarking framework.

Provides infrastructure for systematic comparison of optimizers across
multiple tasks, hyperparameters, and random seeds.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from belavkin_ml.optimizers.belavkin import BelavkinOptimizer, get_belavkin_optimizer
from belavkin_ml.optimizers.baselines import get_baseline_optimizer


class OptimizerBenchmark:
    """
    Framework for benchmarking optimizers on a given task.

    Args:
        model_factory (callable): Function that creates a fresh model
        train_loader (DataLoader): Training data
        test_loader (DataLoader): Test data
        criterion (nn.Module): Loss function
        device (str): Device to use
        max_epochs (int): Maximum training epochs
        eval_every (int): Evaluate every N epochs
        early_stop_patience (int): Early stopping patience (None to disable)
        target_accuracy (float): Target accuracy for convergence timing
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module = None,
        device: str = 'cpu',
        max_epochs: int = 1000,
        eval_every: int = 10,
        early_stop_patience: Optional[int] = None,
        target_accuracy: float = 0.95,
    ):
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.max_epochs = max_epochs
        self.eval_every = eval_every
        self.early_stop_patience = early_stop_patience
        self.target_accuracy = target_accuracy

    def train_with_optimizer(
        self,
        optimizer_name: str,
        optimizer_kwargs: Dict[str, Any],
        seed: int = 42,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train model with specified optimizer and track metrics.

        Args:
            optimizer_name (str): Name of optimizer
            optimizer_kwargs (dict): Optimizer hyperparameters
            seed (int): Random seed
            verbose (bool): Print progress

        Returns:
            results (dict): Training metrics and statistics
        """
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create fresh model
        model = self.model_factory().to(self.device)

        # Create optimizer
        if optimizer_name.lower() in ['belavkin', 'belavkin_sgd', 'belavkin_adam']:
            variant = optimizer_name.lower().replace('belavkin_', '')
            optimizer = get_belavkin_optimizer(
                model.parameters(),
                variant=variant if variant != 'belavkin' else 'default',
                **optimizer_kwargs
            )
        else:
            optimizer = get_baseline_optimizer(
                model.parameters(),
                optimizer_name=optimizer_name,
                **optimizer_kwargs
            )

        # Training metrics
        results = {
            'optimizer': optimizer_name,
            'hyperparams': optimizer_kwargs,
            'seed': seed,
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': [],
            'epochs': [],
            'wall_times': [],
            'convergence_epoch': None,
            'final_train_acc': 0.0,
            'final_test_acc': 0.0,
            'best_test_acc': 0.0,
            'total_time': 0.0,
        }

        start_time = time.time()
        best_test_acc = 0.0
        epochs_without_improvement = 0

        iterator = tqdm(range(self.max_epochs), desc=f"{optimizer_name}") if verbose else range(self.max_epochs)

        for epoch in iterator:
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                predictions = outputs.argmax(dim=1)
                train_correct += (predictions == batch_y).sum().item()
                train_total += len(batch_y)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Evaluation
            if epoch % self.eval_every == 0 or epoch == self.max_epochs - 1:
                test_loss, test_acc = self._evaluate(model)

                results['epochs'].append(epoch)
                results['train_losses'].append(train_loss)
                results['train_accs'].append(train_acc)
                results['test_losses'].append(test_loss)
                results['test_accs'].append(test_acc)
                results['wall_times'].append(time.time() - start_time)

                # Track convergence
                if test_acc >= self.target_accuracy and results['convergence_epoch'] is None:
                    results['convergence_epoch'] = epoch

                # Track best
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += self.eval_every

                # Early stopping
                if self.early_stop_patience and epochs_without_improvement >= self.early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % (self.eval_every * 10) == 0:
                    iterator.set_postfix({
                        'train_acc': f'{train_acc:.3f}',
                        'test_acc': f'{test_acc:.3f}',
                    })

        # Final evaluation
        final_test_loss, final_test_acc = self._evaluate(model)

        results['final_train_acc'] = train_acc
        results['final_test_acc'] = final_test_acc
        results['best_test_acc'] = best_test_acc
        results['total_time'] = time.time() - start_time

        return results

    def _evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set."""
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * len(batch_y)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == batch_y).sum().item()
                total_samples += len(batch_y)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy


def run_optimizer_comparison(
    benchmark: OptimizerBenchmark,
    optimizers: Dict[str, Dict[str, Any]],
    n_seeds: int = 3,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run comparison across multiple optimizers and seeds.

    Args:
        benchmark (OptimizerBenchmark): Benchmark instance
        optimizers (dict): Dict mapping optimizer name to hyperparameters
        n_seeds (int): Number of random seeds to run
        save_dir (Path): Directory to save results
        verbose (bool): Print progress

    Returns:
        all_results (dict): Results for all optimizers and seeds

    Example:
        >>> optimizers = {
        ...     'adam': {'lr': 1e-3},
        ...     'belavkin': {'lr': 1e-3, 'gamma': 1e-4, 'beta': 1e-2},
        ... }
        >>> results = run_optimizer_comparison(benchmark, optimizers, n_seeds=3)
    """
    all_results = {}

    for optimizer_name, optimizer_kwargs in optimizers.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {optimizer_name}")
            print(f"{'='*60}")

        optimizer_results = []

        for seed in range(n_seeds):
            if verbose:
                print(f"\nSeed {seed + 1}/{n_seeds}")

            result = benchmark.train_with_optimizer(
                optimizer_name=optimizer_name,
                optimizer_kwargs=optimizer_kwargs,
                seed=seed,
                verbose=verbose,
            )

            optimizer_results.append(result)

        all_results[optimizer_name] = optimizer_results

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for opt_name, opt_results in all_results.items():
                serializable_results[opt_name] = [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in result.items()}
                    for result in opt_results
                ]
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to {save_dir / 'results.json'}")

    return all_results


def hyperparameter_search(
    benchmark: OptimizerBenchmark,
    optimizer_name: str,
    param_grid: Dict[str, List[Any]],
    n_seeds: int = 3,
    metric: str = 'best_test_acc',
    save_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Grid search over hyperparameters.

    Args:
        benchmark: Benchmark instance
        optimizer_name: Name of optimizer
        param_grid: Dictionary mapping parameter names to lists of values
        n_seeds: Number of seeds per configuration
        metric: Metric to optimize ('best_test_acc', 'convergence_epoch', etc.)
        save_dir: Directory to save results

    Returns:
        best_params, all_results

    Example:
        >>> param_grid = {'lr': [1e-4, 1e-3, 1e-2], 'gamma': [1e-5, 1e-4, 1e-3]}
        >>> best_params, all_results = hyperparameter_search(
        ...     benchmark, 'belavkin', param_grid, n_seeds=3
        ... )
    """
    from itertools import product

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    print(f"Running grid search with {len(all_combinations)} configurations")

    all_results = []
    best_score = -np.inf if 'acc' in metric else np.inf
    best_params = None

    for combo in tqdm(all_combinations, desc="Grid search"):
        params = dict(zip(param_names, combo))

        # Run across seeds
        seed_results = []
        for seed in range(n_seeds):
            result = benchmark.train_with_optimizer(
                optimizer_name=optimizer_name,
                optimizer_kwargs=params,
                seed=seed,
                verbose=False,
            )
            seed_results.append(result)

        # Compute average metric
        metric_values = [r[metric] for r in seed_results if r[metric] is not None]

        if len(metric_values) == 0:
            avg_metric = -np.inf if 'acc' in metric else np.inf
        else:
            avg_metric = np.mean(metric_values)

        summary = {
            'params': params,
            'avg_metric': avg_metric,
            'std_metric': np.std(metric_values) if len(metric_values) > 0 else 0,
            'seed_results': seed_results,
        }

        all_results.append(summary)

        # Update best
        is_better = avg_metric > best_score if 'acc' in metric else avg_metric < best_score
        if is_better:
            best_score = avg_metric
            best_params = params

    print(f"\nBest parameters: {best_params}")
    print(f"Best {metric}: {best_score:.4f}")

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'hyperparameter_search.json', 'w') as f:
            serializable = [
                {
                    'params': r['params'],
                    'avg_metric': float(r['avg_metric']),
                    'std_metric': float(r['std_metric']),
                }
                for r in all_results
            ]
            json.dump({
                'best_params': best_params,
                'best_score': float(best_score),
                'all_results': serializable,
            }, f, indent=2)

    return best_params, all_results
