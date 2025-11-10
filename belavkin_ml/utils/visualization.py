"""
Visualization utilities for analyzing optimizer performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def plot_learning_curves(
    results: Dict[str, List[Dict]],
    metric: str = "test_accs",
    save_path: Optional[Path] = None,
    title: str = "Learning Curves",
    show_std: bool = True,
):
    """
    Plot learning curves for different optimizers.

    Args:
        results: Dictionary mapping optimizer names to list of results
        metric: Metric to plot ("test_accs", "train_accs", "test_losses", etc.)
        save_path: Path to save figure
        title: Plot title
        show_std: Whether to show standard deviation across seeds
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(results))

    for (opt_name, opt_results), color in zip(results.items(), colors):
        # Group by hyperparameters
        configs = {}
        for result in opt_results:
            key = (result['lr'], result.get('gamma'), result.get('beta'))
            if key not in configs:
                configs[key] = []
            configs[key].append(result[metric])

        # Find best config (by final performance)
        best_config = max(configs.items(), key=lambda x: np.mean([vals[-1] for vals in x[1]]))
        curves = best_config[1]

        # Pad curves to same length
        max_len = max(len(c) for c in curves)
        padded_curves = []
        for curve in curves:
            padded = curve + [curve[-1]] * (max_len - len(curve))
            padded_curves.append(padded)

        curves_array = np.array(padded_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)

        epochs = np.arange(len(mean_curve))

        # Plot mean
        ax.plot(epochs, mean_curve, label=opt_name, color=color, linewidth=2)

        # Plot std as shaded region
        if show_std and len(curves) > 1:
            ax.fill_between(
                epochs,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, ax


def plot_optimizer_comparison(
    results: Dict[str, List[Dict]],
    metrics: List[str] = ["best_test_acc", "total_time", "train_test_gap"],
    save_path: Optional[Path] = None,
):
    """
    Create comparison plots for multiple metrics across optimizers.

    Args:
        results: Dictionary mapping optimizer names to list of results
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Aggregate results by optimizer
        opt_names = []
        opt_values = []
        opt_stds = []

        for opt_name, opt_results in results.items():
            # Find best config
            configs = {}
            for result in opt_results:
                key = (result['lr'], result.get('gamma'), result.get('beta'))
                if key not in configs:
                    configs[key] = []
                configs[key].append(result[metric])

            best_config = max(configs.items(), key=lambda x: np.mean(x[1]))
            values = best_config[1]

            opt_names.append(opt_name)
            opt_values.append(np.mean(values))
            opt_stds.append(np.std(values))

        # Create bar plot
        x = np.arange(len(opt_names))
        bars = ax.bar(x, opt_values, yerr=opt_stds, capsize=5, alpha=0.7)

        # Color bars
        colors = sns.color_palette("husl", len(opt_names))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(opt_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_hyperparameter_sensitivity(
    results: List[Dict],
    param_name: str,
    metric: str = "best_test_acc",
    save_path: Optional[Path] = None,
):
    """
    Plot sensitivity to a specific hyperparameter.

    Args:
        results: List of results for a single optimizer
        param_name: Name of hyperparameter ('lr', 'gamma', 'beta')
        metric: Metric to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group by parameter value
    param_values = {}
    for result in results:
        param_val = result[param_name]
        if param_val not in param_values:
            param_values[param_val] = []
        param_values[param_val].append(result[metric])

    # Sort by parameter value
    sorted_params = sorted(param_values.items())

    x_vals = [p[0] for p in sorted_params]
    y_means = [np.mean(p[1]) for p in sorted_params]
    y_stds = [np.std(p[1]) for p in sorted_params]

    ax.errorbar(x_vals, y_means, yerr=y_stds, marker='o', markersize=8,
                linewidth=2, capsize=5, capthick=2)

    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Sensitivity to {param_name}")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_convergence_speed(
    results: Dict[str, List[Dict]],
    target_acc: float = 0.95,
    save_path: Optional[Path] = None,
):
    """
    Plot steps to reach target accuracy for each optimizer.

    Args:
        results: Dictionary mapping optimizer names to list of results
        target_acc: Target accuracy threshold
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    opt_names = []
    steps_means = []
    steps_stds = []

    for opt_name, opt_results in results.items():
        # Find best config
        configs = {}
        for result in opt_results:
            key = (result['lr'], result.get('gamma'), result.get('beta'))
            if key not in configs:
                configs[key] = []

            steps = result['steps_to_target'].get(target_acc)
            if steps is not None:
                configs[key].append(steps)

        if not configs:
            continue

        # Find config with lowest mean steps
        best_config = min(configs.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))

        if best_config[1]:
            opt_names.append(opt_name)
            steps_means.append(np.mean(best_config[1]))
            steps_stds.append(np.std(best_config[1]))

    if opt_names:
        x = np.arange(len(opt_names))
        bars = ax.bar(x, steps_means, yerr=steps_stds, capsize=5, alpha=0.7)

        colors = sns.color_palette("husl", len(opt_names))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(opt_names, rotation=45, ha='right')
        ax.set_ylabel("Steps to Target")
        ax.set_title(f"Convergence Speed (Target: {target_acc:.1%} accuracy)")
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def create_analysis_report(
    results: Dict[str, List[Dict]],
    save_dir: Path,
    experiment_name: str = "optimizer_benchmark",
):
    """
    Create a comprehensive analysis report with multiple visualizations.

    Args:
        results: Dictionary mapping optimizer names to list of results
        save_dir: Directory to save all figures
        experiment_name: Name for the experiment
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating analysis report in {save_dir}...")

    # 1. Learning curves
    print("  - Generating learning curves...")
    plot_learning_curves(
        results,
        metric="test_accs",
        save_path=save_dir / f"{experiment_name}_learning_curves.png",
        title="Test Accuracy vs Epoch",
    )

    plot_learning_curves(
        results,
        metric="test_losses",
        save_path=save_dir / f"{experiment_name}_loss_curves.png",
        title="Test Loss vs Epoch",
    )

    # 2. Optimizer comparison
    print("  - Generating comparison plots...")
    plot_optimizer_comparison(
        results,
        metrics=["best_test_acc", "total_time", "train_test_gap"],
        save_path=save_dir / f"{experiment_name}_comparison.png",
    )

    # 3. Convergence speed
    print("  - Generating convergence analysis...")
    for target in [0.90, 0.95, 0.99]:
        plot_convergence_speed(
            results,
            target_acc=target,
            save_path=save_dir / f"{experiment_name}_convergence_{int(target*100)}.png",
        )

    # 4. Hyperparameter sensitivity (for Belavkin)
    print("  - Generating hyperparameter sensitivity plots...")
    for opt_name in ["belavkin", "adaptive_belavkin"]:
        if opt_name in results:
            for param in ["lr", "gamma", "beta"]:
                plot_hyperparameter_sensitivity(
                    results[opt_name],
                    param_name=param,
                    save_path=save_dir / f"{experiment_name}_{opt_name}_{param}_sensitivity.png",
                )

    print(f"Analysis report complete! Saved to {save_dir}")

    plt.close('all')
