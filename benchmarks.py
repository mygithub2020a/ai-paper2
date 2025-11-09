"""
Comprehensive benchmarking framework for optimizer comparison.

This module implements training loops, evaluation metrics, and result collection
for comparing the Belavkin Optimizer against standard baselines.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from optimizer import create_optimizer
from datasets import (
    create_dataset,
    SimpleNNModel,
    DeepNNModel,
)


class BenchmarkRunner:
    """Runs optimization benchmarks for different optimizers and datasets."""

    def __init__(
        self,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Args:
            device: Device to run on ('cpu' or 'cuda')
            verbose: Whether to print progress
        """
        self.device = device
        self.verbose = verbose
        self.results = {}

    def train_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataset,
        num_epochs: int = 100,
        batch_size: int = 32,
        loss_fn=None,
    ) -> Dict[str, List[float]]:
        """
        Train a model with a given optimizer.

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            dataset: Dataset instance
            num_epochs: Number of training epochs
            batch_size: Batch size
            loss_fn: Loss function (default: MSELoss)

        Returns:
            Dictionary with training metrics
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        model = model.to(self.device)
        loss_fn = loss_fn.to(self.device)

        losses = []
        times = []
        start_time = time.time()

        X, y = dataset.get_full_data()
        X = X.to(self.device)
        y = y.to(self.device)

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            epoch_start = time.time()

            # Shuffle data
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            epoch_time = time.time() - epoch_start
            times.append(epoch_time)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        total_time = time.time() - start_time

        return {
            "losses": losses,
            "times": times,
            "total_time": total_time,
            "final_loss": losses[-1],
            "min_loss": min(losses),
        }

    def benchmark_optimizer(
        self,
        optimizer_name: str,
        optimizer_kwargs: Dict[str, Any],
        dataset_name: str,
        dataset_kwargs: Dict[str, Any],
        model_class,
        model_kwargs: Dict[str, Any],
        num_epochs: int = 100,
        batch_size: int = 32,
        num_runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Benchmark a single optimizer on a single dataset.

        Args:
            optimizer_name: Name of optimizer
            optimizer_kwargs: Hyperparameters for optimizer
            dataset_name: Name of dataset
            dataset_kwargs: Parameters for dataset
            model_class: Model class to use
            model_kwargs: Model parameters
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with benchmark results
        """
        print(
            f"\n[{optimizer_name.upper()}] Benchmarking on {dataset_name} dataset..."
        )

        all_losses = []
        all_final_losses = []
        all_times = []

        for run in range(num_runs):
            if self.verbose:
                print(f"  Run {run + 1}/{num_runs}")

            # Create fresh instances
            dataset = create_dataset(dataset_name, **dataset_kwargs)
            model = model_class(**model_kwargs)
            optimizer = create_optimizer(
                model.parameters(),
                optimizer_name,
                **optimizer_kwargs,
            )

            # Train
            metrics = self.train_model(
                model,
                optimizer,
                dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

            all_losses.append(metrics["losses"])
            all_final_losses.append(metrics["final_loss"])
            all_times.append(metrics["total_time"])

        # Aggregate results
        losses_array = np.array(all_losses)
        mean_losses = losses_array.mean(axis=0)
        std_losses = losses_array.std(axis=0)

        return {
            "optimizer": optimizer_name,
            "dataset": dataset_name,
            "mean_losses": mean_losses.tolist(),
            "std_losses": std_losses.tolist(),
            "final_loss": np.mean(all_final_losses),
            "final_loss_std": np.std(all_final_losses),
            "total_time": np.mean(all_times),
            "total_time_std": np.std(all_times),
            "all_final_losses": all_final_losses,
        }

    def run_suite(
        self,
        optimizers: Dict[str, Dict[str, Any]],
        datasets: Dict[str, Dict[str, Any]],
        model_specs: Dict[str, Tuple],
        num_epochs: int = 100,
        batch_size: int = 32,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a complete benchmarking suite.

        Args:
            optimizers: Dict of optimizer configs {name: kwargs}
            datasets: Dict of dataset configs {name: kwargs}
            model_specs: Dict of model specs {name: (model_class, kwargs)}
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_runs: Number of runs for averaging

        Returns:
            Complete results dictionary
        """
        print("=" * 70)
        print("BELAVKIN OPTIMIZER BENCHMARK SUITE")
        print("=" * 70)

        results = {}

        for dataset_name, dataset_kwargs in datasets.items():
            print(f"\n{'=' * 70}")
            print(f"DATASET: {dataset_name.upper()}")
            print(f"{'=' * 70}")

            dataset_results = {}

            # Determine input/output dims based on dataset
            if "modular_composition" in dataset_name:
                input_dim = 3
            else:
                input_dim = 2
            output_dim = 1

            for model_name, (model_class, model_base_kwargs) in model_specs.items():
                print(f"\n--- Model: {model_name} ---")

                model_kwargs = {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    **model_base_kwargs,
                }

                model_results = {}

                for opt_name, opt_kwargs in optimizers.items():
                    result = self.benchmark_optimizer(
                        optimizer_name=opt_name,
                        optimizer_kwargs=opt_kwargs,
                        dataset_name=dataset_name,
                        dataset_kwargs=dataset_kwargs,
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        num_runs=num_runs,
                    )

                    model_results[opt_name] = result

                dataset_results[model_name] = model_results

            results[dataset_name] = dataset_results

        self.results = results
        return results

    def print_summary(self, results: Optional[Dict] = None):
        """Print a summary of benchmark results."""
        if results is None:
            results = self.results

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        for dataset_name, dataset_results in results.items():
            print(f"\n{'=' * 70}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 70}")

            for model_name, model_results in dataset_results.items():
                print(f"\n{model_name}:")
                print(f"  {'Optimizer':<20} {'Final Loss':<15} {'Time (s)':<12}")
                print(f"  {'-' * 47}")

                # Sort by final loss
                sorted_results = sorted(
                    model_results.items(),
                    key=lambda x: x[1]["final_loss"],
                )

                for opt_name, metrics in sorted_results:
                    final_loss = metrics["final_loss"]
                    final_loss_std = metrics["final_loss_std"]
                    total_time = metrics["total_time"]

                    print(
                        f"  {opt_name:<20} "
                        f"{final_loss:.6f}±{final_loss_std:.6f}  "
                        f"{total_time:.3f}±{metrics['total_time_std']:.3f}"
                    )


def run_ablation_study(
    base_config: Dict[str, Any],
    param_variations: Dict[str, List[Any]],
    num_epochs: int = 100,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Run ablation study on Belavkin Optimizer hyperparameters.

    Args:
        base_config: Base configuration for Belavkin
        param_variations: Dict of parameter variations {param_name: [values]}
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Ablation study results
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: BELAVKIN OPTIMIZER")
    print("=" * 70)

    runner = BenchmarkRunner()
    ablation_results = {}

    dataset = create_dataset("modular_arithmetic", num_samples=500)
    model_class = SimpleNNModel

    for param_name, values in param_variations.items():
        print(f"\nAblating parameter: {param_name}")
        print(f"  Values: {values}")

        param_results = []

        for value in values:
            config = base_config.copy()
            config[param_name] = value

            model = model_class(input_dim=2, output_dim=1)
            optimizer = create_optimizer(
                model.parameters(),
                "belavkin",
                **config,
            )

            metrics = runner.train_model(
                model,
                optimizer,
                dataset,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )

            param_results.append(
                {
                    "value": value,
                    "final_loss": metrics["final_loss"],
                    "min_loss": metrics["min_loss"],
                }
            )

            print(f"    {param_name}={value}: final_loss={metrics['final_loss']:.6f}")

        ablation_results[param_name] = param_results

    return ablation_results


if __name__ == "__main__":
    # Example usage
    print("Running example benchmark...")

    runner = BenchmarkRunner(verbose=False)

    # Define configurations
    optimizers = {
        "belavkin": {"lr": 0.01, "gamma": 0.1, "beta": 0.01},
        "adam": {"lr": 0.01},
        "sgd": {"lr": 0.01},
        "rmsprop": {"lr": 0.01},
    }

    datasets = {
        "modular_arithmetic": {"modulus": 113, "num_samples": 500},
    }

    model_specs = {
        "simple": (SimpleNNModel, {}),
    }

    # Run suite
    results = runner.run_suite(
        optimizers=optimizers,
        datasets=datasets,
        model_specs=model_specs,
        num_epochs=50,
        batch_size=32,
        num_runs=2,
    )

    runner.print_summary()
