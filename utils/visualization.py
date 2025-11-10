"""
Visualization and Analysis Utilities

This module provides tools for visualizing and analyzing experimental results
for both Track 1 (optimizer) and Track 2 (RL) experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Optional
import matplotlib.patches as mpatches


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_optimizer_comparison(
    results: List[Dict],
    metric: str = 'test_accuracy',
    save_path: Optional[str] = None,
):
    """
    Plot learning curves comparing different optimizers.

    Args:
        results: List of experiment results from OptimizerBenchmark
        metric: Metric to plot ('test_accuracy', 'train_loss', etc.)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by optimizer
    by_optimizer = {}
    for r in results:
        opt_name = r['optimizer']
        if opt_name not in by_optimizer:
            by_optimizer[opt_name] = []
        by_optimizer[opt_name].append(r)

    # Plot each optimizer
    colors = plt.cm.tab10(np.linspace(0, 1, len(by_optimizer)))

    for (opt_name, runs), color in zip(by_optimizer.items(), colors):
        # Extract histories
        histories = [r['history'][metric] for r in runs]

        # Pad to same length
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h)

        # Compute mean and std
        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)

        # Plot
        epochs = np.arange(1, len(mean_curve) + 1)
        ax.plot(epochs, mean_curve, label=opt_name, color=color, linewidth=2)
        ax.fill_between(
            epochs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Optimizer Comparison: {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_convergence_analysis(
    results: List[Dict],
    targets: List[float] = [0.90, 0.95, 0.99],
    save_path: Optional[str] = None,
):
    """
    Plot convergence speed (epochs to reach target accuracy).

    Args:
        results: Experiment results
        targets: Target accuracy thresholds
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by optimizer
    by_optimizer = {}
    for r in results:
        opt_name = r['optimizer']
        if opt_name not in by_optimizer:
            by_optimizer[opt_name] = []
        by_optimizer[opt_name].append(r)

    # Prepare data
    data = []
    for opt_name, runs in by_optimizer.items():
        for target in targets:
            key = f'epochs_to_{int(target*100)}'
            values = [r[key] for r in runs if r.get(key) is not None]
            if values:
                data.append({
                    'Optimizer': opt_name,
                    'Target': f'{target*100:.0f}%',
                    'Epochs': np.mean(values),
                    'Std': np.std(values),
                })

    df = pd.DataFrame(data)

    # Plot grouped bar chart
    optimizers = df['Optimizer'].unique()
    x = np.arange(len(targets))
    width = 0.15
    multiplier = 0

    for optimizer in optimizers:
        subset = df[df['Optimizer'] == optimizer]
        offset = width * multiplier
        ax.bar(
            x + offset,
            subset['Epochs'],
            width,
            label=optimizer,
            yerr=subset['Std'],
            capsize=3,
        )
        multiplier += 1

    ax.set_xlabel('Target Accuracy')
    ax.set_ylabel('Epochs to Reach Target')
    ax.set_title('Convergence Speed Comparison')
    ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels([f'{t*100:.0f}%' for t in targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_hyperparameter_sensitivity(
    results: List[Dict],
    optimizer_name: str,
    param_name: str,
    metric: str = 'best_test_accuracy',
    save_path: Optional[str] = None,
):
    """
    Plot sensitivity to hyperparameter values.

    Args:
        results: Experiment results
        optimizer_name: Name of optimizer to analyze
        param_name: Name of hyperparameter (e.g., 'gamma', 'beta')
        metric: Metric to plot
        save_path: Path to save figure
    """
    # Filter results
    filtered = [
        r for r in results
        if r['optimizer'] == optimizer_name and param_name in r['optimizer_kwargs']
    ]

    if not filtered:
        print(f"No results found for {optimizer_name} with parameter {param_name}")
        return

    # Group by parameter value
    by_param = {}
    for r in filtered:
        param_val = r['optimizer_kwargs'][param_name]
        if param_val not in by_param:
            by_param[param_val] = []
        by_param[param_val].append(r[metric])

    # Sort by parameter value
    param_values = sorted(by_param.keys())
    means = [np.mean(by_param[v]) for v in param_values]
    stds = [np.std(by_param[v]) for v in param_values]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(param_values, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{optimizer_name}: Sensitivity to {param_name}')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_rl_training_curves(
    results: Dict,
    save_path: Optional[str] = None,
):
    """
    Plot RL training curves (episode rewards over time).

    Args:
        results: Training history from BelavkinRLTrainer
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Episode rewards
    rewards = results['episode_rewards']
    episodes = np.arange(1, len(rewards) + 1)

    # Smooth curve using moving average
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], smoothed, linewidth=2, label='Smoothed (MA-20)')

    ax1.plot(episodes, rewards, alpha=0.3, linewidth=0.5, label='Raw')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode lengths
    lengths = results['episode_lengths']
    ax2.plot(episodes, lengths, linewidth=1)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress: Episode Lengths')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_summary_table(
    results: List[Dict],
    metrics: List[str] = ['best_test_accuracy', 'final_test_accuracy', 'epochs_to_95'],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create summary table of experimental results.

    Args:
        results: Experiment results
        metrics: Metrics to include
        save_path: Path to save CSV

    Returns:
        Summary DataFrame
    """
    # Group by optimizer and hyperparameters
    by_config = {}
    for r in results:
        key = (r['optimizer'], tuple(sorted(r['optimizer_kwargs'].items())))
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)

    # Compute statistics
    summary_data = []
    for (opt_name, params), runs in by_config.items():
        row = {
            'Optimizer': opt_name,
            'n_runs': len(runs),
        }

        # Add hyperparameters
        for k, v in dict(params).items():
            row[k] = v

        # Add metrics
        for metric in metrics:
            values = [r[metric] for r in runs if r.get(metric) is not None]
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Sort by best accuracy
    if 'best_test_accuracy_mean' in df.columns:
        df = df.sort_values('best_test_accuracy_mean', ascending=False)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to {save_path}")

    return df


def load_and_visualize_results(
    results_path: str,
    output_dir: str = 'figures',
):
    """
    Load results from JSON and create all visualizations.

    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save figures
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} experimental results from {results_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Learning curves
    plot_optimizer_comparison(
        results,
        metric='test_accuracy',
        save_path=f'{output_dir}/learning_curves_test_acc.png',
    )

    plot_optimizer_comparison(
        results,
        metric='train_loss',
        save_path=f'{output_dir}/learning_curves_train_loss.png',
    )

    # 2. Convergence analysis
    plot_convergence_analysis(
        results,
        save_path=f'{output_dir}/convergence_analysis.png',
    )

    # 3. Summary table
    summary = create_summary_table(
        results,
        save_path=f'{output_dir}/summary_table.csv',
    )

    print("\nSummary Table (Top 5):")
    print(summary.head())

    print(f"\nAll figures saved to {output_dir}/")
