"""
Training and benchmarking infrastructure.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from tqdm import tqdm
import time

from .utils import MetricsTracker, compute_accuracy, compute_gradient_norm, get_device


class Trainer:
    """
    Trainer class for running experiments with different optimizers.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        track_grad_norm: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device is not None else get_device()
        self.track_grad_norm = track_grad_norm

        self.model.to(self.device)

        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'grad_norm': [],
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = MetricsTracker()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Track gradient norm
            grad_norm = None
            if self.track_grad_norm:
                grad_norm = compute_gradient_norm(self.model)

            # Optimizer step
            self.optimizer.step()

            # Compute accuracy
            accuracy = compute_accuracy(output, target)

            # Update metrics
            metrics.update(loss.item(), accuracy, grad_norm)

        return metrics.get_metrics()

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        metrics = MetricsTracker()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                accuracy = compute_accuracy(output, target)

                metrics.update(loss.item(), accuracy)

        return metrics.get_metrics()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
            early_stopping_patience: Stop if validation loss doesn't improve for N epochs

        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        iterator = range(num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            # Train
            train_metrics = self.train_epoch(train_loader)

            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            if self.track_grad_norm:
                self.train_history['grad_norm'].append(train_metrics['grad_norm'])

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['accuracy'].append(val_metrics['accuracy'])

                # Early stopping
                if early_stopping_patience is not None:
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"\nEpoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"\nEpoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.4f}"
                    )

        return {
            'train': self.train_history,
            'val': self.val_history if val_loader is not None else None,
        }


class BenchmarkRunner:
    """
    Run benchmarks comparing multiple optimizers.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        optimizer_configs: Dict[str, Dict[str, Any]],
        criterion: nn.Module = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """
        Args:
            model_factory: Function that creates a new model instance
            optimizer_configs: Dict mapping optimizer names to config dicts
                Example: {
                    'adam': {'class': torch.optim.Adam, 'lr': 0.001},
                    'belavkin': {'class': BelavkinOptimizer, 'lr': 0.001, 'gamma': 1e-4}
                }
            criterion: Loss function (defaults to CrossEntropyLoss)
            device: Device to use
            seed: Random seed
        """
        self.model_factory = model_factory
        self.optimizer_configs = optimizer_configs
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else get_device()
        self.seed = seed

        self.results = {}

    def run_benchmark(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        num_runs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark for all optimizers.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs per run
            num_runs: Number of independent runs per optimizer
            verbose: Whether to print progress

        Returns:
            Dictionary containing results for all optimizers
        """
        results = {}

        for opt_name, opt_config in self.optimizer_configs.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking {opt_name}")
                print(f"{'='*60}")

            opt_results = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'grad_norm': [],
                'training_time': [],
            }

            for run in range(num_runs):
                if verbose and num_runs > 1:
                    print(f"\nRun {run + 1}/{num_runs}")

                # Set seed for reproducibility
                torch.manual_seed(self.seed + run)
                np.random.seed(self.seed + run)

                # Create fresh model
                model = self.model_factory()

                # Create optimizer
                opt_class = opt_config['class']
                opt_params = {k: v for k, v in opt_config.items() if k != 'class'}
                optimizer = opt_class(model.parameters(), **opt_params)

                # Create trainer
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    criterion=self.criterion,
                    device=self.device,
                    track_grad_norm=True,
                )

                # Train
                start_time = time.time()
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    verbose=verbose and num_runs == 1,
                )
                training_time = time.time() - start_time

                # Store results
                opt_results['train_loss'].append(history['train']['loss'])
                opt_results['train_accuracy'].append(history['train']['accuracy'])
                opt_results['grad_norm'].append(history['train']['grad_norm'])
                opt_results['training_time'].append(training_time)

                if history['val'] is not None:
                    opt_results['val_loss'].append(history['val']['loss'])
                    opt_results['val_accuracy'].append(history['val']['accuracy'])

            # Compute statistics across runs
            results[opt_name] = self._compute_statistics(opt_results)

            if verbose:
                self._print_summary(opt_name, results[opt_name])

        self.results = results
        return results

    def _compute_statistics(self, opt_results: Dict[str, List]) -> Dict[str, Any]:
        """Compute mean and std across multiple runs."""
        stats = {}

        for key, values in opt_results.items():
            if len(values) > 0 and isinstance(values[0], list):
                # Time series data (loss, accuracy curves)
                values_array = np.array(values)
                stats[f'{key}_mean'] = values_array.mean(axis=0)
                stats[f'{key}_std'] = values_array.std(axis=0)
                stats[f'{key}_all'] = values
            else:
                # Scalar data (training time)
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)

        return stats

    def _print_summary(self, opt_name: str, results: Dict[str, Any]):
        """Print summary statistics for an optimizer."""
        print(f"\nSummary for {opt_name}:")
        print(f"  Final Train Loss: {results['train_loss_mean'][-1]:.4f} ± {results['train_loss_std'][-1]:.4f}")
        print(f"  Final Train Acc:  {results['train_accuracy_mean'][-1]:.4f} ± {results['train_accuracy_std'][-1]:.4f}")

        if 'val_loss_mean' in results:
            print(f"  Final Val Loss:   {results['val_loss_mean'][-1]:.4f} ± {results['val_loss_std'][-1]:.4f}")
            print(f"  Final Val Acc:    {results['val_accuracy_mean'][-1]:.4f} ± {results['val_accuracy_std'][-1]:.4f}")

        print(f"  Training Time:    {results['training_time_mean']:.2f}s ± {results['training_time_std']:.2f}s")

    def get_comparison_table(self) -> str:
        """Generate a comparison table of all optimizers."""
        if not self.results:
            return "No results available. Run benchmark first."

        # Header
        table = "\n" + "="*80 + "\n"
        table += f"{'Optimizer':<15} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}\n"
        table += "="*80 + "\n"

        # Rows
        for opt_name, results in self.results.items():
            train_loss = f"{results['train_loss_mean'][-1]:.4f}±{results['train_loss_std'][-1]:.4f}"
            train_acc = f"{results['train_accuracy_mean'][-1]:.4f}±{results['train_accuracy_std'][-1]:.4f}"

            if 'val_loss_mean' in results:
                val_loss = f"{results['val_loss_mean'][-1]:.4f}±{results['val_loss_std'][-1]:.4f}"
                val_acc = f"{results['val_accuracy_mean'][-1]:.4f}±{results['val_accuracy_std'][-1]:.4f}"
            else:
                val_loss = "N/A"
                val_acc = "N/A"

            table += f"{opt_name:<15} {train_loss:<15} {train_acc:<15} {val_loss:<15} {val_acc:<15}\n"

        table += "="*80 + "\n"
        return table
