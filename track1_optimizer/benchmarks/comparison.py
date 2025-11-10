"""
Optimizer comparison and benchmarking utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Callable, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from .trainer import train_model, compute_convergence_metrics


class OptimizerComparison:
    """
    Run and compare multiple optimizers on a task.

    Example:
        >>> comparison = OptimizerComparison(
        ...     model_factory=lambda: ModularMLP(),
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
        >>> comparison.add_optimizer("Belavkin", BelavkinOptimizer, lr=1e-3, gamma=1e-4)
        >>> comparison.add_optimizer("Adam", torch.optim.Adam, lr=1e-3)
        >>> results = comparison.run(num_epochs=100, num_seeds=3)
        >>> comparison.plot_results()
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize comparison.

        Args:
            model_factory: Function that creates a fresh model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizers = {}
        self.results = {}

    def add_optimizer(
        self,
        name: str,
        optimizer_class: type,
        **optimizer_kwargs
    ):
        """
        Add an optimizer to the comparison.

        Args:
            name: Name for this optimizer configuration
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            **optimizer_kwargs: Arguments to pass to optimizer constructor
        """
        self.optimizers[name] = {
            "class": optimizer_class,
            "kwargs": optimizer_kwargs,
        }

    def run(
        self,
        num_epochs: int = 100,
        num_seeds: int = 3,
        log_interval: int = 10,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all optimizers with multiple random seeds.

        Args:
            num_epochs: Number of training epochs
            num_seeds: Number of random seeds to try
            log_interval: Logging frequency
            early_stopping_patience: Early stopping patience (None to disable)

        Returns:
            Dictionary mapping optimizer names to list of results (one per seed)
        """
        self.results = {}

        for opt_name, opt_config in self.optimizers.items():
            print(f"\n{'='*60}")
            print(f"Running: {opt_name}")
            print(f"{'='*60}\n")

            seed_results = []

            for seed in range(num_seeds):
                print(f"\n--- Seed {seed + 1}/{num_seeds} ---")

                # Set seed for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create fresh model
                model = self.model_factory()

                # Create optimizer
                optimizer = opt_config["class"](
                    model.parameters(),
                    **opt_config["kwargs"]
                )

                # Train
                history = train_model(
                    model=model,
                    optimizer=optimizer,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=num_epochs,
                    device=self.device,
                    log_interval=log_interval,
                    early_stopping_patience=early_stopping_patience,
                )

                # Compute metrics
                metrics = compute_convergence_metrics(history)

                seed_results.append({
                    "seed": seed,
                    "history": history,
                    "metrics": metrics,
                })

            self.results[opt_name] = seed_results

        return self.results

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics (mean ± std) across seeds.

        Returns:
            DataFrame with summary statistics for each optimizer
        """
        summary_data = []

        for opt_name, seed_results in self.results.items():
            metrics_list = [r["metrics"] for r in seed_results]

            summary = {
                "optimizer": opt_name,
            }

            # Compute mean and std for each metric
            for metric_name in metrics_list[0].keys():
                values = [m[metric_name] for m in metrics_list if m[metric_name] is not None]

                if values:
                    summary[f"{metric_name}_mean"] = np.mean(values)
                    summary[f"{metric_name}_std"] = np.std(values)
                else:
                    summary[f"{metric_name}_mean"] = None
                    summary[f"{metric_name}_std"] = None

            summary_data.append(summary)

        return pd.DataFrame(summary_data)

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot comparison results: training curves and metrics.

        Args:
            save_path: Path to save figure (if None, just display)
        """
        if not self.results:
            print("No results to plot. Run comparison first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Validation Accuracy over Epochs
        ax = axes[0, 0]
        for opt_name, seed_results in self.results.items():
            for result in seed_results:
                history = result["history"]
                ax.plot(
                    history["val_acc"],
                    alpha=0.3,
                    color=self._get_color(opt_name),
                )
            # Plot mean
            mean_val_acc = np.mean([r["history"]["val_acc"] for r in seed_results], axis=0)
            ax.plot(
                mean_val_acc,
                label=opt_name,
                linewidth=2,
                color=self._get_color(opt_name),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Training Loss over Epochs
        ax = axes[0, 1]
        for opt_name, seed_results in self.results.items():
            mean_train_loss = np.mean([r["history"]["train_loss"] for r in seed_results], axis=0)
            ax.plot(
                mean_train_loss,
                label=opt_name,
                linewidth=2,
                color=self._get_color(opt_name),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Plot 3: Best Validation Accuracy (bar plot)
        ax = axes[1, 0]
        summary = self.get_summary_statistics()
        if "best_val_acc_mean" in summary.columns:
            x = range(len(summary))
            ax.bar(
                x,
                summary["best_val_acc_mean"],
                yerr=summary["best_val_acc_std"],
                capsize=5,
                color=[self._get_color(opt) for opt in summary["optimizer"]],
                alpha=0.7,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(summary["optimizer"], rotation=45, ha="right")
            ax.set_ylabel("Best Validation Accuracy (%)")
            ax.set_title("Best Performance")
            ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Convergence Speed (steps to 95%)
        ax = axes[1, 1]
        if "steps_to_95_mean" in summary.columns:
            valid_data = summary[summary["steps_to_95_mean"].notna()]
            if not valid_data.empty:
                x = range(len(valid_data))
                ax.bar(
                    x,
                    valid_data["steps_to_95_mean"],
                    yerr=valid_data["steps_to_95_std"],
                    capsize=5,
                    color=[self._get_color(opt) for opt in valid_data["optimizer"]],
                    alpha=0.7,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(valid_data["optimizer"], rotation=45, ha="right")
                ax.set_ylabel("Epochs to 95% Accuracy")
                ax.set_title("Convergence Speed")
                ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, path: str):
        """
        Save results to JSON file.

        Args:
            path: Path to save results
        """
        # Convert results to serializable format
        serializable_results = {}
        for opt_name, seed_results in self.results.items():
            serializable_results[opt_name] = [
                {
                    "seed": r["seed"],
                    "metrics": r["metrics"],
                    "history": {
                        k: [float(v) for v in vals] if isinstance(vals, list) else vals
                        for k, vals in r["history"].items()
                    },
                }
                for r in seed_results
            ]

        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {path}")

    def _get_color(self, optimizer_name: str) -> str:
        """Get consistent color for optimizer."""
        colors = {
            "Belavkin": "#e74c3c",  # Red
            "Adam": "#3498db",  # Blue
            "SGD": "#2ecc71",  # Green
            "RMSprop": "#f39c12",  # Orange
            "AdamW": "#9b59b6",  # Purple
        }
        return colors.get(optimizer_name, "#95a5a6")  # Gray as default


def run_benchmark(
    task_name: str,
    model_factory: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    num_seeds: int = 3,
    output_dir: str = "./results",
) -> OptimizerComparison:
    """
    Run a complete benchmark comparison.

    Args:
        task_name: Name of the task (for saving results)
        model_factory: Function creating model instances
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs per run
        num_seeds: Number of random seeds
        output_dir: Directory to save results

    Returns:
        OptimizerComparison object with results
    """
    from ..belavkin_optimizer import BelavkinOptimizer

    comparison = OptimizerComparison(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Add optimizers with standard hyperparameters
    comparison.add_optimizer("Belavkin", BelavkinOptimizer, lr=1e-3, gamma=1e-4, beta=1e-2)
    comparison.add_optimizer("Adam", torch.optim.Adam, lr=1e-3)
    comparison.add_optimizer("SGD", torch.optim.SGD, lr=1e-2, momentum=0.9)
    comparison.add_optimizer("RMSprop", torch.optim.RMSprop, lr=1e-3, alpha=0.99)
    comparison.add_optimizer("AdamW", torch.optim.AdamW, lr=1e-3, weight_decay=0.01)

    # Run comparison
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {task_name}")
    print(f"{'='*60}\n")

    results = comparison.run(num_epochs=num_epochs, num_seeds=num_seeds)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison.save_results(str(output_path / f"{task_name}_results.json"))
    comparison.plot_results(save_path=str(output_path / f"{task_name}_plot.png"))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    summary = comparison.get_summary_statistics()
    print(summary.to_string(index=False))

    return comparison
