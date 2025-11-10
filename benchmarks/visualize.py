"""
Visualization utilities for benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

from .utils import load_results

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_loss_curves(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    title: str = "Training Loss Comparison",
):
    """
    Plot loss curves for all optimizers.

    Args:
        results: Dictionary of benchmark results
        output_path: Path to save figure (optional)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for opt_name, opt_results in results.items():
        epochs = np.arange(len(opt_results['train_loss_mean']))

        # Train loss
        ax1.plot(epochs, opt_results['train_loss_mean'], label=opt_name, linewidth=2)
        ax1.fill_between(
            epochs,
            opt_results['train_loss_mean'] - opt_results['train_loss_std'],
            opt_results['train_loss_mean'] + opt_results['train_loss_std'],
            alpha=0.2,
        )

        # Val loss
        if 'val_loss_mean' in opt_results:
            ax2.plot(epochs, opt_results['val_loss_mean'], label=opt_name, linewidth=2)
            ax2.fill_between(
                epochs,
                opt_results['val_loss_mean'] - opt_results['val_loss_std'],
                opt_results['val_loss_mean'] + opt_results['val_loss_std'],
                alpha=0.2,
            )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_accuracy_curves(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    title: str = "Accuracy Comparison",
):
    """Plot accuracy curves for all optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for opt_name, opt_results in results.items():
        epochs = np.arange(len(opt_results['train_accuracy_mean']))

        # Train accuracy
        ax1.plot(epochs, opt_results['train_accuracy_mean'], label=opt_name, linewidth=2)
        ax1.fill_between(
            epochs,
            opt_results['train_accuracy_mean'] - opt_results['train_accuracy_std'],
            opt_results['train_accuracy_mean'] + opt_results['train_accuracy_std'],
            alpha=0.2,
        )

        # Val accuracy
        if 'val_accuracy_mean' in opt_results:
            ax2.plot(epochs, opt_results['val_accuracy_mean'], label=opt_name, linewidth=2)
            ax2.fill_between(
                epochs,
                opt_results['val_accuracy_mean'] - opt_results['val_accuracy_std'],
                opt_results['val_accuracy_mean'] + opt_results['val_accuracy_std'],
                alpha=0.2,
            )

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def plot_gradient_norms(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
    title: str = "Gradient Norm Comparison",
):
    """Plot gradient norm curves."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for opt_name, opt_results in results.items():
        if 'grad_norm_mean' in opt_results:
            epochs = np.arange(len(opt_results['grad_norm_mean']))
            ax.plot(epochs, opt_results['grad_norm_mean'], label=opt_name, linewidth=2)
            ax.fill_between(
                epochs,
                opt_results['grad_norm_mean'] - opt_results['grad_norm_std'],
                opt_results['grad_norm_mean'] + opt_results['grad_norm_std'],
                alpha=0.2,
            )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


def create_comparison_table(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> str:
    """Create a formatted comparison table."""
    table = "\n" + "="*100 + "\n"
    table += f"{'Optimizer':<15} {'Train Loss':<20} {'Train Acc':<20} {'Val Loss':<20} {'Val Acc':<20}\n"
    table += "="*100 + "\n"

    for opt_name, opt_results in results.items():
        train_loss = f"{opt_results['train_loss_mean'][-1]:.4f} ± {opt_results['train_loss_std'][-1]:.4f}"
        train_acc = f"{opt_results['train_accuracy_mean'][-1]:.4f} ± {opt_results['train_accuracy_std'][-1]:.4f}"

        if 'val_loss_mean' in opt_results:
            val_loss = f"{opt_results['val_loss_mean'][-1]:.4f} ± {opt_results['val_loss_std'][-1]:.4f}"
            val_acc = f"{opt_results['val_accuracy_mean'][-1]:.4f} ± {opt_results['val_accuracy_std'][-1]:.4f}"
        else:
            val_loss = "N/A"
            val_acc = "N/A"

        table += f"{opt_name:<15} {train_loss:<20} {train_acc:<20} {val_loss:<20} {val_acc:<20}\n"

    table += "="*100 + "\n"

    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Saved table to {output_path}")

    return table


def visualize_all_results(results_dir: str, output_dir: str):
    """
    Load all results and create comprehensive visualizations.

    Args:
        results_dir: Directory containing .pkl result files
        output_dir: Directory to save visualizations
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = list(results_dir.glob('*.pkl'))

    print(f"Found {len(result_files)} result files")

    for result_file in result_files:
        print(f"\nProcessing {result_file.name}...")

        # Load results
        results = load_results(str(result_file))

        # Create task name
        task_name = result_file.stem.replace('_results', '')

        # Generate plots
        plot_loss_curves(
            results,
            output_path=str(output_dir / f'{task_name}_loss.png'),
            title=f'{task_name.replace("_", " ").title()} - Loss Curves',
        )

        plot_accuracy_curves(
            results,
            output_path=str(output_dir / f'{task_name}_accuracy.png'),
            title=f'{task_name.replace("_", " ").title()} - Accuracy Curves',
        )

        plot_gradient_norms(
            results,
            output_path=str(output_dir / f'{task_name}_gradnorm.png'),
            title=f'{task_name.replace("_", " ").title()} - Gradient Norms',
        )

        # Create table
        table = create_comparison_table(
            results,
            output_path=str(output_dir / f'{task_name}_table.txt'),
        )
        print(table)

    print(f"\nAll visualizations saved to {output_dir}")


def plot_ablation_results(
    ablation_results: Dict[str, Dict],
    param_name: str,
    output_path: Optional[str] = None,
):
    """
    Plot ablation study results.

    Args:
        ablation_results: Results from ablation study
        param_name: Name of parameter being ablated
        output_path: Path to save figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    param_values = []
    final_train_loss = []
    final_train_acc = []
    final_val_loss = []
    final_val_acc = []

    for param_val, results in sorted(ablation_results.items()):
        param_values.append(param_val)
        final_train_loss.append(results['train_loss_mean'][-1])
        final_train_acc.append(results['train_accuracy_mean'][-1])

        if 'val_loss_mean' in results:
            final_val_loss.append(results['val_loss_mean'][-1])
            final_val_acc.append(results['val_accuracy_mean'][-1])

    # Plot final metrics vs parameter value
    ax1.plot(param_values, final_train_loss, 'o-', linewidth=2)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Final Training Loss')
    ax1.set_title('Training Loss vs ' + param_name)
    ax1.grid(True)
    ax1.set_xscale('log')

    ax2.plot(param_values, final_train_acc, 'o-', linewidth=2)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Final Training Accuracy')
    ax2.set_title('Training Accuracy vs ' + param_name)
    ax2.grid(True)
    ax2.set_xscale('log')

    if final_val_loss:
        ax3.plot(param_values, final_val_loss, 'o-', linewidth=2)
        ax3.set_xlabel(param_name)
        ax3.set_ylabel('Final Validation Loss')
        ax3.set_title('Validation Loss vs ' + param_name)
        ax3.grid(True)
        ax3.set_xscale('log')

        ax4.plot(param_values, final_val_acc, 'o-', linewidth=2)
        ax4.set_xlabel(param_name)
        ax4.set_ylabel('Final Validation Accuracy')
        ax4.set_title('Validation Accuracy vs ' + param_name)
        ax4.grid(True)
        ax4.set_xscale('log')

    fig.suptitle(f'Ablation Study: {param_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ablation plot to {output_path}")

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing result files')
    parser.add_argument('--output-dir', type=str, default='paper/figures',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    visualize_all_results(args.results_dir, args.output_dir)
