"""
Visualize scaling behavior of optimizers.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")


def plot_scaling_curves(results_path, output_dir='figures'):
    """Plot how accuracy scales with problem size."""
    print("Creating scaling visualizations...")

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Extract scaling data
    primes = []
    optimizers = set()

    for p_key in data.keys():
        p = int(p_key.split('=')[1])
        primes.append(p)
        optimizers.update(data[p_key]['summary'].keys())

    primes = sorted(primes)
    optimizers = sorted(optimizers)

    # Create main scaling plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Best accuracy vs problem size
    for opt in optimizers:
        accuracies = []
        for p in primes:
            p_key = f'p={p}'
            if opt in data[p_key]['summary']:
                accuracies.append(data[p_key]['summary'][opt]['best_accuracy'])
            else:
                accuracies.append(np.nan)

        ax1.plot(primes, accuracies, 'o-', linewidth=2, markersize=8, label=opt)

    ax1.set_xlabel('Prime p (problem size = p²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling: Accuracy vs Problem Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative performance (Adam as baseline)
    if 'adam' in optimizers:
        adam_accs = []
        for p in primes:
            p_key = f'p={p}'
            if 'adam' in data[p_key]['summary']:
                adam_accs.append(data[p_key]['summary']['adam']['best_accuracy'])
            else:
                adam_accs.append(1.0)

        for opt in optimizers:
            if opt == 'adam':
                continue

            relative = []
            for i, p in enumerate(primes):
                p_key = f'p={p}'
                if opt in data[p_key]['summary']:
                    opt_acc = data[p_key]['summary'][opt]['best_accuracy']
                    relative.append((opt_acc - adam_accs[i]) * 100)
                else:
                    relative.append(np.nan)

            ax2.plot(primes, relative, 'o-', linewidth=2, markersize=8, label=opt)

        ax2.set_xlabel('Prime p (problem size = p²)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Gap vs Adam (percentage points)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Performance vs Adam', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/scaling_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

    # Create detailed comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table
    table_data = [['Optimizer'] + [f'p={p}' for p in primes]]

    for opt in optimizers:
        row = [opt]
        for p in primes:
            p_key = f'p={p}'
            if opt in data[p_key]['summary']:
                acc = data[p_key]['summary'][opt]['best_accuracy']
                row.append(f'{acc:.3f}')
            else:
                row.append('N/A')
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Scaling Results: Best Accuracy by Problem Size',
              fontsize=14, fontweight='bold', pad=20)

    output_path = f'{output_dir}/scaling_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

    print("\n✓ Scaling visualizations created")


def analyze_quantum_components(results_path):
    """Analyze whether quantum components help at any scale."""
    print("\nAnalyzing quantum components across scales...")

    with open(results_path, 'r') as f:
        data = json.load(f)

    primes = sorted([int(k.split('=')[1]) for k in data.keys()])

    print(f"\n{'Prime':>6s} {'No Quantum':>12s} {'With Quantum':>12s} {'Difference':>12s} {'Effect':>10s}")
    print("-" * 60)

    helps_count = 0
    hurts_count = 0

    for p in primes:
        p_key = f'p={p}'

        if 'belavkin' in data[p_key]['summary'] and 'belavkin_with_quantum' in data[p_key]['summary']:
            no_q = data[p_key]['summary']['belavkin']['best_accuracy']
            with_q = data[p_key]['summary']['belavkin_with_quantum']['best_accuracy']
            diff = with_q - no_q

            if diff > 0.01:
                effect = "✓ Helps"
                helps_count += 1
            elif diff < -0.01:
                effect = "✗ Hurts"
                hurts_count += 1
            else:
                effect = "~ Neutral"

            print(f"{p:6d} {no_q:12.4f} {with_q:12.4f} {diff:+12.4f} {effect:>10s}")

    print("\n" + "="*60)
    print("QUANTUM COMPONENTS SUMMARY")
    print("="*60)
    print(f"Helps: {helps_count}/{len(primes)} cases")
    print(f"Hurts: {hurts_count}/{len(primes)} cases")

    if hurts_count > helps_count:
        print("\n✗ Quantum components generally HURT performance")
    elif helps_count > hurts_count:
        print("\n✓ Quantum components generally HELP performance")
    else:
        print("\n~ Quantum components have NEUTRAL effect")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = 'results/scalability/scaling_test.json'

    if os.path.exists(results_path):
        plot_scaling_curves(results_path)
        analyze_quantum_components(results_path)
    else:
        print(f"Results file not found: {results_path}")
        print("Run scalability_test.py first")
