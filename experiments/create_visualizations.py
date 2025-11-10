"""
Create visualizations from experimental results.
"""

import sys
import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_results(path):
    """Load results from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison_bar(results_path, output_path):
    """Create bar plot comparing optimizers."""
    print(f"Creating comparison bar plot...")

    # Load results
    data = load_results(results_path)
    summary = data['summary']

    # Extract data
    optimizers = [s['optimizer'] for s in summary]
    best_acc = [s['best_accuracy'] for s in summary]
    mean_acc = [s['mean_accuracy'] for s in summary]
    std_acc = [s['std_accuracy'] for s in summary]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(optimizers))
    width = 0.35

    # Bar plot
    bars = ax.bar(x, best_acc, width, label='Best', alpha=0.8)
    ax.errorbar(x + width, mean_acc, yerr=std_acc, fmt='o', capsize=5,
                label='Mean ± Std', markersize=8)

    # Styling
    ax.set_xlabel('Optimizer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Optimizer Comparison: Modular Addition (p=11)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line at 100%
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')

    # Color bars
    colors = ['green', 'green', 'green', 'orange', 'red']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def plot_learning_curves(results_path, output_path):
    """Plot learning curves for each optimizer."""
    print(f"Creating learning curves...")

    data = load_results(results_path)
    all_results = data['all_results']

    # Group by optimizer
    by_optimizer = {}
    for r in all_results:
        opt = r['optimizer']
        if opt not in by_optimizer:
            by_optimizer[opt] = []
        by_optimizer[opt].append(r)

    # Create figure with subplots
    n_opts = len(by_optimizer)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (opt_name, runs) in enumerate(by_optimizer.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot each run
        for run in runs:
            history = run['history']
            test_acc = history['test_accuracy']
            epochs = np.arange(1, len(test_acc) + 1)
            ax.plot(epochs, test_acc, alpha=0.3, linewidth=1)

        # Plot mean
        max_len = max(len(r['history']['test_accuracy']) for r in runs)
        all_curves = []
        for r in runs:
            curve = r['history']['test_accuracy']
            # Pad if needed
            if len(curve) < max_len:
                curve = curve + [curve[-1]] * (max_len - len(curve))
            all_curves.append(curve)

        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        epochs = np.arange(1, len(mean_curve) + 1)

        ax.plot(epochs, mean_curve, linewidth=3, label='Mean', color='blue')
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color='blue')

        ax.set_title(f'{opt_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)

    # Hide extra subplots
    for idx in range(len(by_optimizer), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Learning Curves: All Optimizers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def plot_best_configs(results_path, output_path):
    """Plot best configuration for each optimizer."""
    print(f"Creating best configurations plot...")

    data = load_results(results_path)
    summary = data['summary']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create text summary
    y_pos = 0.9
    for i, s in enumerate(summary):
        opt = s['optimizer']
        best_acc = s['best_accuracy']
        params = s['best_params']

        # Format params
        param_str = ", ".join([f"{k}={v:.0e}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in params.items()])

        # Color based on performance
        color = 'green' if best_acc >= 0.95 else 'orange' if best_acc >= 0.7 else 'red'

        text = f"{i+1}. {opt:15s} | Accuracy: {best_acc:.4f}\n    {param_str}"
        ax.text(0.05, y_pos, text, transform=ax.transAxes,
                fontsize=11, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

        y_pos -= 0.18

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Best Configurations for Each Optimizer', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def create_summary_table_image(results_path, output_path):
    """Create table image of results."""
    print(f"Creating summary table...")

    data = load_results(results_path)
    summary = data['summary']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Rank', 'Optimizer', 'Best Acc.', 'Mean Acc.', 'Std', 'Best Hyperparameters'])

    for i, s in enumerate(summary, 1):
        params_str = ", ".join([f"{k}={v:.0e}" if isinstance(v, float) and v < 0.01
                                else f"{k}={v}"
                                for k, v in s['best_params'].items()])
        table_data.append([
            str(i),
            s['optimizer'],
            f"{s['best_accuracy']:.4f}",
            f"{s['mean_accuracy']:.4f}",
            f"{s['std_accuracy']:.4f}",
            params_str
        ])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.06, 0.15, 0.10, 0.10, 0.08, 0.51])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Color rows based on performance
    for i in range(1, len(table_data)):
        best_acc = float(table_data[i][2])
        color = '#90EE90' if best_acc >= 0.95 else '#FFE4B5' if best_acc >= 0.7 else '#FFB6C1'
        for j in range(6):
            table[(i, j)].set_facecolor(color)

    plt.title('Optimizer Comparison: Complete Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    results_path = 'results/final/comparison.json'
    output_dir = 'figures'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    try:
        plot_comparison_bar(results_path, f'{output_dir}/comparison_bar.png')
        plot_learning_curves(results_path, f'{output_dir}/learning_curves.png')
        plot_best_configs(results_path, f'{output_dir}/best_configs.png')
        create_summary_table_image(results_path, f'{output_dir}/summary_table.png')

        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
        print("="*70)
        print(f"\nFigures saved to: {output_dir}/")
        print("\nGenerated files:")
        print(f"  - comparison_bar.png")
        print(f"  - learning_curves.png")
        print(f"  - best_configs.png")
        print(f"  - summary_table.png")

    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
