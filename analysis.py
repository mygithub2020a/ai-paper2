"""
Analysis and visualization tools for benchmark results.

This module provides functions for analyzing benchmark results, generating
visualizations, and producing publication-ready figures.
"""

import json
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

# Use non-interactive backend for server environments
matplotlib.use("Agg")


class BenchmarkAnalyzer:
    """Analyzes and visualizes benchmark results."""

    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_loss_curves(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Plot loss curves for all optimizer and dataset combinations.

        Args:
            results: Benchmark results dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        plot_idx = 1
        num_datasets = len(results)
        num_models = max(
            len(dataset_results)
            for dataset_results in results.values()
        )

        for dataset_name, dataset_results in results.items():
            for model_name, model_results in dataset_results.items():
                ax = plt.subplot(num_datasets, num_models, plot_idx)

                for opt_name, metrics in model_results.items():
                    mean_losses = metrics["mean_losses"]
                    std_losses = metrics["std_losses"]

                    epochs = range(1, len(mean_losses) + 1)

                    ax.plot(epochs, mean_losses, label=opt_name, linewidth=2)
                    ax.fill_between(
                        epochs,
                        np.array(mean_losses) - np.array(std_losses),
                        np.array(mean_losses) + np.array(std_losses),
                        alpha=0.2,
                    )

                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel("Loss", fontsize=10)
                ax.set_title(f"{dataset_name} - {model_name}", fontsize=11, fontweight="bold")
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

                plot_idx += 1

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "loss_curves.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")
        plt.close()

    def plot_final_loss_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Create bar plots comparing final losses across optimizers.

        Args:
            results: Benchmark results dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        plot_idx = 1
        num_datasets = len(results)
        num_models = max(
            len(dataset_results)
            for dataset_results in results.values()
        )

        for dataset_name, dataset_results in results.items():
            for model_name, model_results in dataset_results.items():
                ax = plt.subplot(num_datasets, num_models, plot_idx)

                optimizers = []
                final_losses = []
                std_losses = []

                for opt_name, metrics in model_results.items():
                    optimizers.append(opt_name)
                    final_losses.append(metrics["final_loss"])
                    std_losses.append(metrics["final_loss_std"])

                x_pos = np.arange(len(optimizers))
                colors = ["#FF6B6B" if "belavkin" in opt else "#4ECDC4"
                          for opt in optimizers]

                ax.bar(x_pos, final_losses, yerr=std_losses, capsize=5,
                       color=colors, alpha=0.7, edgecolor="black")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(optimizers, rotation=45, ha="right")
                ax.set_ylabel("Final Loss", fontsize=10)
                ax.set_title(f"{dataset_name} - {model_name}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")
                ax.set_yscale("log")

                plot_idx += 1

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "final_loss_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Final loss comparison saved to {save_path}")
        plt.close()

    def create_results_table(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a summary table of results.

        Args:
            results: Benchmark results dictionary
            save_path: Path to save CSV

        Returns:
            DataFrame with summary statistics
        """
        rows = []

        for dataset_name, dataset_results in results.items():
            for model_name, model_results in dataset_results.items():
                for opt_name, metrics in model_results.items():
                    rows.append({
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Optimizer": opt_name,
                        "Final Loss": f"{metrics['final_loss']:.6f}",
                        "Final Loss Std": f"{metrics['final_loss_std']:.6f}",
                        "Min Loss": f"{min(metrics['mean_losses']):.6f}",
                        "Total Time (s)": f"{metrics['total_time']:.3f}",
                    })

        df = pd.DataFrame(rows)

        if save_path is None:
            save_path = self.output_dir / "results_summary.csv"

        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")

        return df

    def plot_convergence_speed(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot convergence speed (epochs to reach target loss).

        Args:
            results: Benchmark results dictionary
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Aggregate results across datasets and models
        optimizer_final_losses = {}
        optimizer_convergence_epochs = {}

        for dataset_name, dataset_results in results.items():
            for model_name, model_results in dataset_results.items():
                for opt_name, metrics in model_results.items():
                    if opt_name not in optimizer_final_losses:
                        optimizer_final_losses[opt_name] = []
                        optimizer_convergence_epochs[opt_name] = []

                    optimizer_final_losses[opt_name].append(metrics["final_loss"])

                    # Find epoch to reach 10x improvement from initial
                    mean_losses = metrics["mean_losses"]
                    target = mean_losses[0] / 10
                    epochs_to_target = next(
                        (i for i, loss in enumerate(mean_losses) if loss < target),
                        len(mean_losses)
                    )
                    optimizer_convergence_epochs[opt_name].append(epochs_to_target)

        # Plot final losses
        ax = axes[0]
        optimizers = list(optimizer_final_losses.keys())
        final_losses_mean = [
            np.mean(optimizer_final_losses[opt]) for opt in optimizers
        ]
        final_losses_std = [
            np.std(optimizer_final_losses[opt]) for opt in optimizers
        ]

        colors = ["#FF6B6B" if "belavkin" in opt else "#4ECDC4"
                  for opt in optimizers]

        ax.bar(range(len(optimizers)), final_losses_mean, yerr=final_losses_std,
               capsize=5, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(optimizers)))
        ax.set_xticklabels(optimizers, rotation=45, ha="right")
        ax.set_ylabel("Average Final Loss")
        ax.set_title("Final Loss Comparison")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")

        # Plot convergence speed
        ax = axes[1]
        convergence_mean = [
            np.mean(optimizer_convergence_epochs[opt]) for opt in optimizers
        ]
        convergence_std = [
            np.std(optimizer_convergence_epochs[opt]) for opt in optimizers
        ]

        ax.bar(range(len(optimizers)), convergence_mean, yerr=convergence_std,
               capsize=5, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(optimizers)))
        ax.set_xticklabels(optimizers, rotation=45, ha="right")
        ax.set_ylabel("Epochs to 10x Improvement")
        ax.set_title("Convergence Speed Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "convergence_speed.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Convergence speed plot saved to {save_path}")
        plt.close()

    def plot_ablation_study(
        self,
        ablation_results: Dict[str, List[Dict]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Plot ablation study results.

        Args:
            ablation_results: Ablation study results
            save_path: Path to save figure
            figsize: Figure size
        """
        num_params = len(ablation_results)
        fig, axes = plt.subplots(
            (num_params + 1) // 2,
            2 if num_params > 1 else 1,
            figsize=figsize,
        )

        if num_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (param_name, results) in enumerate(ablation_results.items()):
            ax = axes[idx]

            values = [r["value"] for r in results]
            final_losses = [r["final_loss"] for r in results]
            min_losses = [r["min_loss"] for r in results]

            x_pos = np.arange(len(values))

            ax.plot(x_pos, final_losses, "o-", label="Final Loss", linewidth=2, markersize=8)
            ax.plot(x_pos, min_losses, "s--", label="Min Loss", linewidth=2, markersize=8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(values)
            ax.set_xlabel(f"{param_name}", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.set_title(f"Ablation: {param_name}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Remove extra subplots
        for idx in range(num_params, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "ablation_study.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Ablation study plot saved to {save_path}")
        plt.close()

    def save_results(self, results: Dict, filepath: str):
        """Save results to pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """Load results from pickle file."""
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        return results


def generate_analysis_report(results: Dict, ablation_results: Dict, output_dir: str = "results"):
    """Generate all analysis plots and tables."""
    analyzer = BenchmarkAnalyzer(output_dir=output_dir)

    print("\nGenerating analysis plots and tables...")

    # Generate all visualizations
    analyzer.plot_loss_curves(results)
    analyzer.plot_final_loss_comparison(results)
    analyzer.plot_convergence_speed(results)
    analyzer.create_results_table(results)

    if ablation_results:
        analyzer.plot_ablation_study(ablation_results)

    print(f"\nAll results saved to {output_dir}/")

    return analyzer


if __name__ == "__main__":
    # Example usage (requires pre-existing results)
    print("Analysis module loaded successfully")
