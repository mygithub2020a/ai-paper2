"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    results: Dict[str, Any],
    metrics: List[str] = ['train_accs', 'test_accs'],
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves for a single optimizer run.

    Args:
        results (dict): Results dictionary from benchmark
        metrics (list): Metrics to plot
        save_path (Path): Path to save figure
        title (str): Plot title

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    epochs = results['epochs']

    for ax, metric in zip(axes, metrics):
        values = results[metric]
        ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Training Curves - {results['optimizer']}", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_optimizer_comparison(
    all_results: Dict[str, List[Dict[str, Any]]],
    metric: str = 'test_accs',
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_std: bool = True,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot comparison of multiple optimizers.

    Args:
        all_results (dict): Results from run_optimizer_comparison
        metric (str): Metric to plot
        save_path (Path): Path to save figure
        title (str): Plot title
        show_std (bool): Show standard deviation band
        log_scale (bool): Use log scale for y-axis

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (optimizer_name, optimizer_results), color in zip(all_results.items(), colors):
        # Collect all runs
        all_curves = []
        max_len = 0

        for result in optimizer_results:
            epochs = result['epochs']
            values = result[metric]
            all_curves.append((epochs, values))
            max_len = max(max_len, len(epochs))

        # Interpolate to common epoch grid
        common_epochs = all_curves[0][0]  # Use first run's epochs as reference
        interpolated_curves = []

        for epochs, values in all_curves:
            # Simple interpolation to match common epochs
            if len(epochs) == len(common_epochs):
                interpolated_curves.append(values)
            else:
                # Pad with last value if shorter
                padded = list(values)
                while len(padded) < len(common_epochs):
                    padded.append(padded[-1])
                interpolated_curves.append(padded[:len(common_epochs)])

        interpolated_curves = np.array(interpolated_curves)

        # Compute mean and std
        mean_curve = interpolated_curves.mean(axis=0)
        std_curve = interpolated_curves.std(axis=0)

        # Plot
        ax.plot(common_epochs, mean_curve, label=optimizer_name, linewidth=2, color=color)

        if show_std:
            ax.fill_between(
                common_epochs,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Optimizer Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_convergence_analysis(
    all_results: Dict[str, List[Dict[str, Any]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot convergence analysis: time to target accuracy, final performance, etc.

    Args:
        all_results (dict): Results from run_optimizer_comparison
        save_path (Path): Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    optimizer_names = list(all_results.keys())

    # 1. Final test accuracy
    final_accs = []
    final_accs_std = []

    for optimizer_name in optimizer_names:
        accs = [r['final_test_acc'] for r in all_results[optimizer_name]]
        final_accs.append(np.mean(accs))
        final_accs_std.append(np.std(accs))

    axes[0].bar(optimizer_names, final_accs, yerr=final_accs_std, capsize=5, alpha=0.7)
    axes[0].set_ylabel('Final Test Accuracy')
    axes[0].set_title('Final Performance')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Convergence epoch (time to target accuracy)
    convergence_epochs = []
    convergence_epochs_std = []

    for optimizer_name in optimizer_names:
        epochs = [r['convergence_epoch'] for r in all_results[optimizer_name]
                  if r['convergence_epoch'] is not None]

        if len(epochs) > 0:
            convergence_epochs.append(np.mean(epochs))
            convergence_epochs_std.append(np.std(epochs))
        else:
            convergence_epochs.append(np.nan)
            convergence_epochs_std.append(0)

    axes[1].bar(optimizer_names, convergence_epochs, yerr=convergence_epochs_std,
                capsize=5, alpha=0.7, color='orange')
    axes[1].set_ylabel('Epochs to Target Accuracy')
    axes[1].set_title('Convergence Speed')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 3. Wall-clock time
    total_times = []
    total_times_std = []

    for optimizer_name in optimizer_names:
        times = [r['total_time'] for r in all_results[optimizer_name]]
        total_times.append(np.mean(times))
        total_times_std.append(np.std(times))

    axes[2].bar(optimizer_names, total_times, yerr=total_times_std,
                capsize=5, alpha=0.7, color='green')
    axes[2].set_ylabel('Total Time (seconds)')
    axes[2].set_title('Computational Cost')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_hyperparameter_heatmap(
    search_results: List[Dict[str, Any]],
    param1: str,
    param2: str,
    metric: str = 'avg_metric',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot heatmap of hyperparameter search results.

    Args:
        search_results: Results from hyperparameter_search
        param1: First parameter name (x-axis)
        param2: Second parameter name (y-axis)
        metric: Metric to visualize
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    # Extract unique parameter values
    param1_values = sorted(set(r['params'][param1] for r in search_results))
    param2_values = sorted(set(r['params'][param2] for r in search_results))

    # Create grid
    grid = np.full((len(param2_values), len(param1_values)), np.nan)

    for result in search_results:
        i = param2_values.index(result['params'][param2])
        j = param1_values.index(result['params'][param1])
        grid[i, j] = result[metric]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(grid, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(param1_values)))
    ax.set_yticks(range(len(param2_values)))
    ax.set_xticklabels([f'{v:.1e}' for v in param1_values], rotation=45)
    ax.set_yticklabels([f'{v:.1e}' for v in param2_values])
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title(f'Hyperparameter Search: {metric}')

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
