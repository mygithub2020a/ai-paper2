"""
Scalability test: Belavkin optimizer on larger modular arithmetic problems.

Tests on increasing problem sizes to see if performance changes.
"""

import sys
import os
import torch
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import create_modular_task
from experiments.benchmark import OptimizerBenchmark


def test_scaling(primes=[11, 23, 47, 97], n_epochs=300, n_seeds=3):
    """Test optimizer performance across different problem sizes."""
    print("\n" + "="*70)
    print("SCALABILITY TEST: MODULAR ARITHMETIC AT DIFFERENT SCALES")
    print("="*70)
    print(f"\nTesting primes: {primes}")
    print(f"Epochs: {n_epochs}, Seeds: {n_seeds}")
    print()

    all_results = {}

    for p in primes:
        print("\n" + "="*70)
        print(f"TESTING WITH p={p} (dataset size: {p*p})")
        print("="*70)

        # Create task
        model_fn = lambda: __import__('experiments.synthetic_tasks', fromlist=['ModularArithmeticModel']).ModularArithmeticModel(
            p=p, hidden_dim=128, n_layers=2, operation='add'
        )

        from experiments.synthetic_tasks import ModularArithmeticDataset
        from torch.utils.data import DataLoader

        dataset = ModularArithmeticDataset(p=p, operation='add', train_fraction=0.5)

        dataset.set_train()
        train_loader = DataLoader(dataset, batch_size=min(512, p*p//4), shuffle=True)

        dataset.set_test()
        test_loader = DataLoader(dataset, batch_size=min(512, p*p//4), shuffle=False)

        # Create benchmark
        benchmark = OptimizerBenchmark(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            device='cpu'
        )

        # Test key configurations
        configs = {
            'adam': {
                'lr': [1e-4, 3e-4, 1e-3],
            },
            'sgd': {
                'lr': [1e-3, 1e-2],
                'momentum': [0.9],
            },
            'belavkin': {
                'lr': [1e-3, 1e-2, 3e-2],
                'gamma': [0],  # Best from previous experiments
                'beta': [0],
            },
            'belavkin_with_quantum': {
                'lr': [1e-2],
                'gamma': [1e-5, 1e-4],  # Test quantum components
                'beta': [1e-3, 1e-2],
            }
        }

        # Adjust epochs based on problem size
        epochs_for_p = min(n_epochs, 500 if p > 50 else 300)

        print(f"\nRunning {epochs_for_p} epochs...")
        results = benchmark.grid_search(
            optimizer_configs=configs,
            n_seeds=n_seeds,
            n_epochs=epochs_for_p,
            log_interval=max(50, epochs_for_p//6)
        )

        # Analyze results for this prime
        by_optimizer = {}
        for r in results:
            opt_name = r['optimizer']
            if opt_name not in by_optimizer:
                by_optimizer[opt_name] = []
            by_optimizer[opt_name].append(r)

        summary = {}
        for opt_name, runs in by_optimizer.items():
            best_run = max(runs, key=lambda x: x['best_test_accuracy'])
            mean_acc = np.mean([r['best_test_accuracy'] for r in runs])
            std_acc = np.std([r['best_test_accuracy'] for r in runs])

            summary[opt_name] = {
                'best_accuracy': best_run['best_test_accuracy'],
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'best_params': best_run['optimizer_kwargs'],
                'final_epoch': best_run['total_epochs'],
                'time_per_epoch': best_run['time_per_epoch'],
            }

        all_results[f'p={p}'] = {
            'summary': summary,
            'full_results': results
        }

        # Print summary for this prime
        print(f"\n{'='*70}")
        print(f"RESULTS FOR p={p}")
        print(f"{'='*70}")

        sorted_opts = sorted(summary.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
        for i, (opt_name, stats) in enumerate(sorted_opts, 1):
            print(f"{i}. {opt_name:25s} | Best: {stats['best_accuracy']:.4f} | "
                  f"Mean: {stats['mean_accuracy']:.4f}±{stats['std_accuracy']:.4f}")

    # Save all results
    os.makedirs('results/scalability', exist_ok=True)
    with open('results/scalability/scaling_test.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create comparison table across scales
    print("\n" + "="*70)
    print("SCALING COMPARISON: BEST ACCURACY BY OPTIMIZER AND PRIME")
    print("="*70)
    print()

    # Get all optimizers
    all_optimizers = set()
    for p_results in all_results.values():
        all_optimizers.update(p_results['summary'].keys())

    # Print header
    print(f"{'Optimizer':25s}", end='')
    for p in primes:
        print(f" | p={p:3d}", end='')
    print()
    print("-" * 70)

    # Print results for each optimizer
    for opt in sorted(all_optimizers):
        print(f"{opt:25s}", end='')
        for p in primes:
            p_key = f'p={p}'
            if opt in all_results[p_key]['summary']:
                acc = all_results[p_key]['summary'][opt]['best_accuracy']
                print(f" | {acc:5.3f}", end='')
            else:
                print(f" |   N/A", end='')
        print()

    # Analyze trends
    print("\n" + "="*70)
    print("ANALYSIS: PERFORMANCE TRENDS")
    print("="*70)

    for opt in ['adam', 'belavkin', 'belavkin_with_quantum']:
        if opt not in all_optimizers:
            continue

        print(f"\n{opt}:")
        accuracies = []
        for p in primes:
            p_key = f'p={p}'
            if opt in all_results[p_key]['summary']:
                acc = all_results[p_key]['summary'][opt]['best_accuracy']
                accuracies.append(acc)
                print(f"  p={p:3d}: {acc:.4f}")

        if len(accuracies) > 1:
            trend = "improving" if accuracies[-1] > accuracies[0] else "declining"
            print(f"  Trend: {trend}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Check if Belavkin ever beats Adam
    belavkin_wins = []
    adam_wins = []

    for p in primes:
        p_key = f'p={p}'
        if 'adam' in all_results[p_key]['summary'] and 'belavkin' in all_results[p_key]['summary']:
            adam_acc = all_results[p_key]['summary']['adam']['best_accuracy']
            bel_acc = all_results[p_key]['summary']['belavkin']['best_accuracy']

            if bel_acc > adam_acc:
                belavkin_wins.append(p)
            else:
                adam_wins.append(p)

    print(f"\nAdam wins: {len(adam_wins)}/{len(primes)} problems")
    print(f"Belavkin wins: {len(belavkin_wins)}/{len(primes)} problems")

    if belavkin_wins:
        print(f"Belavkin wins on: p={belavkin_wins}")
    else:
        print("Belavkin NEVER beats Adam across all problem sizes")

    # Check quantum components
    if 'belavkin_with_quantum' in all_optimizers:
        print("\nQuantum components analysis:")
        for p in primes:
            p_key = f'p={p}'
            if 'belavkin' in all_results[p_key]['summary'] and 'belavkin_with_quantum' in all_results[p_key]['summary']:
                no_q = all_results[p_key]['summary']['belavkin']['best_accuracy']
                with_q = all_results[p_key]['summary']['belavkin_with_quantum']['best_accuracy']
                diff = with_q - no_q
                direction = "helps" if diff > 0.01 else "hurts" if diff < -0.01 else "neutral"
                print(f"  p={p}: Without quantum: {no_q:.4f}, With quantum: {with_q:.4f} ({direction})")

    print("\n" + "="*70)
    print("Results saved to results/scalability/scaling_test.json")
    print("="*70)

    return all_results


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # Test on increasing problem sizes
    results = test_scaling(
        primes=[11, 23, 47, 97],
        n_epochs=300,
        n_seeds=3
    )
