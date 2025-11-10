"""
Plotting utilities for visualizing benchmark results.

Generates:
- Learning curves (loss/accuracy vs epoch)
- Time-to-target comparisons
- Optimizer comparison tables
- Ablation study plots
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_logs(log_dir, pattern='*.json'):
    """Load all log files matching pattern."""
    log_files = list(Path(log_dir).glob(pattern))
    logs = {}

    for log_file in log_files:
        with open(log_file, 'r') as f:
            logs[log_file.stem] = json.load(f)

    return logs


def plot_learning_curves(logs, save_dir, metric='test_acc'):
    """Plot learning curves for multiple experiments."""
    # Group by optimizer
    optimizer_logs = {}
    for exp_name, log in logs.items():
        # Parse experiment name: task_optimizer_mXXX_dXX_sXX
        parts = exp_name.split('_')
        if len(parts) >= 2:
            optimizer = parts[1]
            if optimizer not in optimizer_logs:
                optimizer_logs[optimizer] = []
            optimizer_logs[optimizer].append(log)

    # Plot
    fig, ax = plt.subplots()

    for optimizer, exp_logs in optimizer_logs.items():
        # Compute mean and std across seeds
        max_len = max(len(log) for log in exp_logs)
        values = np.zeros((len(exp_logs), max_len))
        values[:] = np.nan

        for i, log in enumerate(exp_logs):
            for j, entry in enumerate(log):
                if j < max_len:
                    values[i, j] = entry[metric]

        epochs = np.arange(1, max_len + 1)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)

        ax.plot(epochs, mean, label=optimizer, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'learning_curves_{metric}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_time_to_target(results_csv, save_dir, target_acc=90.0):
    """Plot time-to-target comparison."""
    df = pd.read_csv(results_csv)

    # Filter out experiments that didn't reach target
    df_reached = df[df['time_to_target'].notna()]

    if len(df_reached) == 0:
        print("No experiments reached target accuracy")
        return

    # Group by optimizer
    summary = df_reached.groupby('optimizer')['time_to_target'].agg(['mean', 'std'])
    summary = summary.sort_values('mean')

    # Plot
    fig, ax = plt.subplots()

    x = np.arange(len(summary))
    ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha='right')
    ax.set_ylabel(f'Time to {target_acc}% Accuracy (seconds)')
    ax.set_title('Time-to-Target Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'time_to_target.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_final_accuracy(results_csv, save_dir):
    """Plot final accuracy comparison."""
    df = pd.read_csv(results_csv)

    # Group by task and optimizer
    summary = df.groupby(['task', 'optimizer'])['best_test_acc'].agg(['mean', 'std'])
    summary = summary.reset_index()

    # Plot
    tasks = summary['task'].unique()
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for i, task in enumerate(tasks):
        task_data = summary[summary['task'] == task].sort_values('mean', ascending=False)

        x = np.arange(len(task_data))
        axes[i].bar(x, task_data['mean'], yerr=task_data['std'], capsize=5, alpha=0.7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(task_data['optimizer'], rotation=45, ha='right')
        axes[i].set_ylabel('Best Test Accuracy (%)')
        axes[i].set_title(f'Task: {task}')
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_ylim([0, 100])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'final_accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_results_table(results_csv, save_dir):
    """Create LaTeX table of results."""
    df = pd.read_csv(results_csv)

    # Group by task and optimizer
    summary = df.groupby(['task', 'optimizer']).agg({
        'best_test_acc': ['mean', 'std'],
        'time_to_target': ['mean', 'std'],
    }).round(2)

    # Save as LaTeX
    latex_table = summary.to_latex()
    save_path = os.path.join(save_dir, 'results_table.tex')
    with open(save_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved: {save_path}")

    # Also save as CSV
    save_path_csv = os.path.join(save_dir, 'results_summary.csv')
    summary.to_csv(save_path_csv)
    print(f"Saved: {save_path_csv}")


def main(args):
    """Generate all plots."""
    os.makedirs(args.save_dir, exist_ok=True)

    # Load logs
    print(f"Loading logs from {args.log_dir}...")
    logs = load_logs(args.log_dir)
    print(f"Loaded {len(logs)} experiment logs")

    if len(logs) == 0:
        print("No logs found!")
        return

    # Plot learning curves
    print("\nGenerating learning curves...")
    plot_learning_curves(logs, args.save_dir, metric='test_acc')
    plot_learning_curves(logs, args.save_dir, metric='test_loss')

    # Check for results CSV
    results_csv = os.path.join(args.log_dir, 'benchmark_results.csv')
    if os.path.exists(results_csv):
        print("\nGenerating comparison plots...")
        plot_time_to_target(results_csv, args.save_dir)
        plot_final_accuracy(results_csv, args.save_dir)
        create_results_table(results_csv, args.save_dir)
    else:
        print(f"\nNo benchmark_results.csv found at {results_csv}")
        print("Run benchmarks first to generate aggregated results")

    print(f"\nAll plots saved to {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot benchmark results')

    parser.add_argument('--log_dir', type=str, default='./results/supervised',
                        help='Directory containing log files')
    parser.add_argument('--save_dir', type=str, default='./belavkin/paper/figs',
                        help='Directory to save plots')

    args = parser.parse_args()
    main(args)
